"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"
    PARENT_COLLECTION_NAME = "knowledge_base_parent"
    CHILD_COLLECTION_NAME = "knowledge_base_child"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._parent_collection = self._client.get_or_create_collection(
            name=self.PARENT_COLLECTION_NAME,
            metadata={"description": "EchoMind RAG 知识库"},
        )
        self._child_collection = self._client.get_or_create_collection(
            name=self.CHILD_COLLECTION_NAME,
            metadata={"description": "EchoMind RAG 知识库子块索引"},
        )
        # 兼容旧调用和旧测试桩
        self._collection = self._child_collection

        # 如果知识库为空，导入默认文档
        if self._child_collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        parent_ids, parent_docs, parent_metas = [], [], []
        child_ids, child_docs, child_metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            parent_chunks = self._build_structured_chunks(title, content)
            child_chunks = self._build_child_chunks(parent_chunks)

            for chunk in parent_chunks:
                parent_ids.append(chunk["parent_id"])
                parent_docs.append(chunk["content"])
                parent_metas.append({
                    "title": chunk["title"],
                    "doc_id": chunk["doc_id"],
                    "parent_id": chunk["parent_id"],
                    "section_title": chunk["section_title"],
                    "heading_path": chunk["heading_path"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                })

            for chunk in child_chunks:
                child_id = hashlib.md5(
                    f"{chunk['parent_id']}_{chunk['child_chunk_index']}_{chunk['content'][:50]}".encode()
                ).hexdigest()
                child_ids.append(child_id)
                child_docs.append(chunk["content"])
                child_metas.append({
                    "title": chunk["title"],
                    "doc_id": chunk["doc_id"],
                    "parent_id": chunk["parent_id"],
                    "section_title": chunk["section_title"],
                    "heading_path": chunk["heading_path"],
                    "chunk_index": chunk["chunk_index"],
                    "child_chunk_index": chunk["child_chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                })

        if parent_ids:
            self._parent_collection.add(ids=parent_ids, documents=parent_docs, metadatas=parent_metas)
        if child_ids:
            self._child_collection.add(ids=child_ids, documents=child_docs, metadatas=child_metas)
            logger.info(f"知识库导入 {len(parent_ids)} 个父块和 {len(child_ids)} 个子块")

        return len(parent_ids) + len(child_ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        child_collection = getattr(self, "_child_collection", self._collection)
        parent_collection = getattr(self, "_parent_collection", None)

        results = child_collection.query(
            query_texts=[query],
            n_results=max(top_k, 5),
        )

        items = []
        if results["documents"] and results["documents"][0]:
            parent_hits: Dict[str, Dict[str, Any]] = {}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                parent_id = meta.get("parent_id")
                score = round(1.0 - dist, 4)
                if parent_id and parent_id not in parent_hits:
                    parent_hits[parent_id] = {
                        "score": score,
                        "matched_child_content": doc,
                        "matched_child_chunk": meta.get("child_chunk_index", meta.get("chunk_index", 0)),
                        "child_meta": meta,
                    }
                elif parent_id and score > parent_hits[parent_id]["score"]:
                    parent_hits[parent_id].update({
                        "score": score,
                        "matched_child_content": doc,
                        "matched_child_chunk": meta.get("child_chunk_index", meta.get("chunk_index", 0)),
                        "child_meta": meta,
                    })
                elif not parent_id:
                    items.append({
                        "title": meta.get("title", ""),
                        "content": doc,
                        "score": score,
                        "chunk": meta.get("chunk_index", 0),
                        "doc_id": meta.get("doc_id", ""),
                        "section_title": meta.get("section_title", meta.get("title", "")),
                        "heading_path": meta.get("heading_path", meta.get("title", "")),
                    })

            if parent_hits and parent_collection is not None:
                parent_results = parent_collection.get(ids=list(parent_hits.keys()))
                parent_docs = parent_results.get("documents", [])
                parent_metas = parent_results.get("metadatas", [])
                parent_ids = parent_results.get("ids", [])

                normalized_docs = self._flatten_result_rows(parent_docs)
                normalized_metas = self._flatten_result_rows(parent_metas)
                normalized_ids = self._flatten_result_rows(parent_ids)

                for parent_id, doc, meta in zip(normalized_ids, normalized_docs, normalized_metas):
                    hit = parent_hits.get(parent_id)
                    if hit is None:
                        continue
                    items.append({
                        "title": meta.get("title", ""),
                        "content": doc,
                        "score": hit["score"],
                        "chunk": meta.get("chunk_index", 0),
                        "doc_id": meta.get("doc_id", ""),
                        "section_title": meta.get("section_title", meta.get("title", "")),
                        "heading_path": meta.get("heading_path", meta.get("title", "")),
                        "matched_child_content": hit["matched_child_content"],
                        "matched_child_chunk": hit["matched_child_chunk"],
                    })

        items.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return items[:top_k]

    @property
    def doc_count(self) -> int:
        parent_collection = getattr(self, "_parent_collection", None)
        child_collection = getattr(self, "_child_collection", self._collection)
        parent_count = parent_collection.count() if parent_collection is not None else 0
        child_count = child_collection.count() if child_collection is not None else 0
        return parent_count + child_count

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return self.search(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
        """段落优先切分，段落过长时按多标点句切，并在相邻块之间保留 overlap。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        if len(normalized) <= chunk_size:
            return [normalized]

        paragraphs = self._split_paragraphs(normalized)
        base_chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            for piece in self._split_paragraph_chunk(paragraph, chunk_size):
                if not current:
                    current = piece
                    continue

                merged = f"{current}\n\n{piece}"
                if len(merged) <= chunk_size:
                    current = merged
                    continue

                base_chunks.append(current)
                current = piece

        if current:
            base_chunks.append(current)

        return self._apply_overlap(base_chunks, overlap)

    def _build_structured_chunks(self, title: str, content: str) -> List[Dict[str, Any]]:
        normalized = self._normalize_text(content)
        if not normalized:
            return []

        doc_id = hashlib.md5(f"{title}:{normalized[:200]}".encode()).hexdigest()
        sections = self._split_sections(title, normalized)
        chunks: List[Dict[str, Any]] = []

        for section in sections:
            section_chunks = self._chunk_text(section["content"])
            for chunk in section_chunks:
                chunk_index = len(chunks)
                chunks.append({
                    "doc_id": doc_id,
                    "parent_id": f"{doc_id}:parent:{chunk_index}",
                    "title": title,
                    "section_title": section["section_title"],
                    "heading_path": section["heading_path"],
                    "content": chunk,
                })

        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk["chunk_index"] = idx
            chunk["total_chunks"] = total
        return chunks

    def _build_child_chunks(self, parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        child_chunks: List[Dict[str, Any]] = []
        for parent in parent_chunks:
            pieces = self._chunk_text(parent["content"], chunk_size=180, overlap=40)
            for child_idx, piece in enumerate(pieces):
                child_chunks.append({
                    "doc_id": parent["doc_id"],
                    "parent_id": parent["parent_id"],
                    "title": parent["title"],
                    "section_title": parent["section_title"],
                    "heading_path": parent["heading_path"],
                    "chunk_index": parent["chunk_index"],
                    "total_chunks": parent["total_chunks"],
                    "child_chunk_index": child_idx,
                    "content": piece,
                })
        return child_chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _split_sections(self, title: str, text: str) -> List[Dict[str, str]]:
        sections: List[Dict[str, str]] = []
        current_heading = title.strip() or "未命名文档"
        current_path = current_heading
        current_lines: List[str] = []

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue

            heading = self._extract_heading(line)
            if heading:
                if current_lines and any(part.strip() for part in current_lines):
                    sections.append({
                        "section_title": current_heading,
                        "heading_path": current_path,
                        "content": "\n".join(current_lines).strip(),
                    })
                    current_lines = []

                current_heading = heading
                current_path = heading if heading == title.strip() else f"{title.strip() or '未命名文档'} > {heading}"
                continue

            current_lines.append(raw_line)

        if current_lines and any(part.strip() for part in current_lines):
            sections.append({
                "section_title": current_heading,
                "heading_path": current_path,
                "content": "\n".join(current_lines).strip(),
            })

        return sections or [{
            "section_title": title.strip() or "未命名文档",
            "heading_path": title.strip() or "未命名文档",
            "content": text,
        }]

    @staticmethod
    def _extract_heading(line: str) -> Optional[str]:
        if not line:
            return None

        markdown = re.match(r"^#{1,6}\s+(.+)$", line)
        if markdown:
            return markdown.group(1).strip()

        if re.match(r"^(第[一二三四五六七八九十百]+[章节部分]|[一二三四五六七八九十]+、|\d+[.)、])", line):
            return line.strip()

        if line.endswith("：") and len(line) <= 30:
            return line[:-1].strip()

        return None

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n\s*\n+", text)
        return [part.strip() for part in parts if part.strip()]

    def _split_paragraph_chunk(self, paragraph: str, chunk_size: int) -> List[str]:
        if len(paragraph) <= chunk_size:
            return [paragraph]

        sentences = self._split_sentences(paragraph)
        pieces: List[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._hard_split(sentence, chunk_size))
                continue

            if not current:
                current = sentence
                continue

            separator = self._sentence_separator(current, sentence)
            candidate = f"{current}{separator}{sentence}"
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                pieces.append(current)
                current = sentence

        if current:
            pieces.append(current)

        return pieces

    def _split_sentences(self, paragraph: str) -> List[str]:
        parts = re.split(r"([。！？!?；;.])", paragraph)
        sentences: List[str] = []
        for i in range(0, len(parts), 2):
            text = parts[i].strip()
            if not text:
                continue
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            sentences.append(f"{text}{punct}")
        return sentences or [paragraph]

    @staticmethod
    def _sentence_separator(current: str, sentence: str) -> str:
        if not current or not sentence:
            return ""
        if current[-1] in "。！？!?；;":
            return ""
        return " "

    def _hard_split(self, text: str, chunk_size: int) -> List[str]:
        pieces: List[str] = []
        remaining = text.strip()
        while remaining:
            pieces.append(remaining[:chunk_size].strip())
            remaining = remaining[chunk_size:].strip()
        return [piece for piece in pieces if piece]

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prefix = merged[-1][-overlap:]
            if prefix and not chunk.startswith(prefix):
                merged.append(f"{prefix}{chunk}")
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _flatten_result_rows(rows: Any) -> List[Any]:
        if not isinstance(rows, list):
            return []
        if rows and isinstance(rows[0], list):
            return rows[0]
        return rows

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
