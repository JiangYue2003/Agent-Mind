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
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
import httpx

logger = logging.getLogger(__name__)

def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    DEFAULT_RECALL_TOP_K = 20
    DEFAULT_RRF_K = 60
    DEFAULT_RERANK_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."
    DEFAULT_RERANK_SCORE_THRESHOLD = 0.25

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        self._child_records: List[Dict[str, Any]] = []
        self._bm25_doc_tokens: List[List[str]] = []
        self._bm25_term_freqs: List[Counter[str]] = []
        self._bm25_doc_freq: Dict[str, int] = {}
        self._bm25_avgdl = 0.0
        self._hybrid_recall_k = self._read_int_env("RAG_HYBRID_RECALL_K", self.DEFAULT_RECALL_TOP_K)
        self._rrf_k = self._read_int_env("RAG_RRF_K", self.DEFAULT_RRF_K)
        self._rerank_url = os.getenv("DASHSCOPE_RERANK_URL", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks").strip()
        self._rerank_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        self._rerank_model = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank").strip() or "qwen3-rerank"
        self._rerank_instruct = os.getenv("DASHSCOPE_RERANK_INSTRUCT", self.DEFAULT_RERANK_INSTRUCT).strip() or self.DEFAULT_RERANK_INSTRUCT
        self._rerank_score_threshold = self._read_float_env("RAG_RERANK_SCORE_THRESHOLD", self.DEFAULT_RERANK_SCORE_THRESHOLD)

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
            metadata={"description": "EchoMind RAG 知识库父块"},
        )
        self._child_collection = self._client.get_or_create_collection(
            name=self.CHILD_COLLECTION_NAME,
            metadata={"description": "EchoMind RAG 知识库子块索引"},
        )
        # 兼容旧调用和旧测试桩
        self._collection = self._child_collection

        # 如果知识库为空，导入默认文档
        if self._child_collection.count() == 0:
            if self._should_auto_seed():
                self._load_default_docs()
        else:
            self._load_child_records_from_collection()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        if not hasattr(self, "_child_records"):
            self._child_records = []
        total_added = 0
        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            metadata = self._build_doc_metadata(doc)

            parent_ids, parent_docs, parent_metas = [], [], []
            child_ids, child_docs, child_metas = [], [], []
            child_records: List[Dict[str, Any]] = []

            parent_chunks = self._build_structured_chunks(title, content, metadata=metadata)
            child_chunks = self._build_child_chunks(parent_chunks)

            for chunk in parent_chunks:
                parent_ids.append(chunk["parent_id"])
                parent_docs.append(chunk["content"])
                parent_metas.append(self._sanitize_metadata({
                    "title": chunk["title"],
                    "doc_id": chunk["doc_id"],
                    "parent_id": chunk["parent_id"],
                    "section_title": chunk["section_title"],
                    "heading_path": chunk["heading_path"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                }))

            for chunk in child_chunks:
                child_id = hashlib.md5(
                    f"{chunk['parent_id']}_{chunk['child_chunk_index']}_{chunk['content'][:50]}".encode()
                ).hexdigest()
                child_ids.append(child_id)
                child_docs.append(chunk["content"])
                child_metas.append(self._sanitize_metadata({
                    "title": chunk["title"],
                    "doc_id": chunk["doc_id"],
                    "parent_id": chunk["parent_id"],
                    "section_title": chunk["section_title"],
                    "heading_path": chunk["heading_path"],
                    "chunk_index": chunk["chunk_index"],
                    "child_chunk_index": chunk["child_chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                }))
                child_records.append({
                    "title": chunk["title"],
                    "content": chunk["content"],
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
                logger.info(f"知识库导入完成: title={title} 父块={len(parent_ids)} 子块={len(child_ids)}")
                self._child_records.extend(child_records)
                total_added += len(parent_ids) + len(child_ids)

        self._rebuild_bm25_index()
        return total_added

    @staticmethod
    def _should_auto_seed() -> bool:
        raw = os.getenv("KNOWLEDGE_AUTO_SEED", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def search(self, query: str, top_k: int = 5, recall_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        混合检索：向量召回 + BM25 召回 → RRF 融合 → qwen3-rerank 重排。

        最终返回精确 child chunk，同时补充所属 parent 内容，便于后续上下文注入。
        """
        resolved_recall_k = max(
            top_k,
            int(recall_k) if recall_k is not None else self._resolve_recall_k(top_k),
        )
        vector_hits = self._vector_recall(query, top_n=resolved_recall_k)
        bm25_hits = self._bm25_recall(query, top_n=resolved_recall_k)
        fused_hits = self._fuse_recall_results(vector_hits, bm25_hits, top_n=resolved_recall_k)
        reranked = self._rerank_candidates(query, fused_hits, top_n=top_k)
        return self._attach_parent_context(reranked)

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
        recall_k = params.get("recall_k")
        return self.search(query, top_k=top_k, recall_k=recall_k)

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

    def _build_structured_chunks(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        normalized = self._normalize_text(content)
        if not normalized:
            return []

        metadata = metadata or {}
        doc_id = str(metadata.get("doc_id") or hashlib.md5(f"{title}:{normalized[:200]}".encode()).hexdigest())
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

    def _resolve_recall_k(self, top_k: int) -> int:
        baseline = max(top_k, getattr(self, "_hybrid_recall_k", self.DEFAULT_RECALL_TOP_K))
        child_count = len(getattr(self, "_child_records", []))
        if child_count > 0:
            return max(top_k, min(baseline, child_count))
        return baseline

    def _vector_recall(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        child_collection = self._get_child_collection()
        if child_collection is None:
            return []

        try:
            results = child_collection.query(
                query_texts=[query],
                n_results=max(top_n, 1),
            )
        except Exception as ex:
            logger.warning(f"向量召回失败: {ex}")
            return []

        docs = self._flatten_result_rows(results.get("documents", []))
        metas = self._flatten_result_rows(results.get("metadatas", []))
        dists = self._flatten_result_rows(results.get("distances", []))

        hits: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not isinstance(meta, dict):
                continue
            score = round(1.0 - float(dist), 4)
            hits.append(self._build_candidate_from_meta(meta, doc, vector_score=score))

        hits.sort(key=lambda item: item.get("vector_score", 0.0), reverse=True)
        return hits[:top_n]

    def _bm25_recall(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        self._ensure_bm25_index()
        records = list(getattr(self, "_child_records", []))
        if not records:
            return []

        query_terms = self._tokenize_for_bm25(query)
        if not query_terms:
            return []

        avgdl = self._get_bm25_avgdl() or 1.0
        term_freqs = getattr(self, "_bm25_term_freqs", [])
        doc_tokens = getattr(self, "_bm25_doc_tokens", [])
        doc_freq = getattr(self, "_bm25_doc_freq", {})
        total_docs = len(records)
        scored: List[Dict[str, Any]] = []

        for idx, record in enumerate(records):
            if idx >= len(term_freqs) or idx >= len(doc_tokens):
                continue
            tf = term_freqs[idx]
            doc_len = max(len(doc_tokens[idx]), 1)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                numerator = freq * (1.5 + 1.0)
                denominator = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / avgdl)
                score += idf * (numerator / denominator)

            if score <= 0:
                continue

            candidate = self._build_candidate_from_meta(record, record.get("content", ""), bm25_score=round(score, 4))
            scored.append(candidate)

        scored.sort(key=lambda item: item.get("bm25_score", 0.0), reverse=True)
        return scored[:top_n]

    def _fuse_recall_results(
        self,
        vector_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}

        def merge_hits(hits: List[Dict[str, Any]], source: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                key = self._candidate_key(hit)
                merged = fused.setdefault(key, dict(hit))
                merged.setdefault("vector_score", 0.0)
                merged.setdefault("bm25_score", 0.0)
                merged.setdefault("rrf_score", 0.0)
                merged["rrf_score"] += 1.0 / (getattr(self, "_rrf_k", self.DEFAULT_RRF_K) + rank)
                if source == "vector":
                    merged["vector_score"] = max(float(merged.get("vector_score", 0.0)), float(hit.get("vector_score", hit.get("score", 0.0))))
                else:
                    merged["bm25_score"] = max(float(merged.get("bm25_score", 0.0)), float(hit.get("bm25_score", hit.get("score", 0.0))))

        merge_hits(vector_hits, "vector")
        merge_hits(bm25_hits, "bm25")

        items = list(fused.values())
        items.sort(
            key=lambda item: (
                item.get("rrf_score", 0.0),
                item.get("vector_score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )
        for item in items:
            item["score"] = round(float(item.get("rrf_score", 0.0)), 4)
        return items[:top_n]

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if len(candidates) <= top_n:
            return self._apply_rerank_scores(candidates)

        api_key = getattr(self, "_rerank_api_key", "").strip()
        if not api_key:
            return self._apply_rerank_scores(candidates[:top_n])

        payload = {
            "model": getattr(self, "_rerank_model", "qwen3-rerank"),
            "query": query,
            "documents": [str(item.get("content", ""))[:4000] for item in candidates],
            "top_n": min(top_n, len(candidates)),
            "instruct": getattr(self, "_rerank_instruct", self.DEFAULT_RERANK_INSTRUCT),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                getattr(self, "_rerank_url", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"),
                headers=headers,
                json=payload,
                timeout=20.0,
            )
            response.raise_for_status()
            body = response.json()
            results = body.get("results", [])
            if not isinstance(results, list) or not results:
                raise ValueError("rerank 响应缺少 results")

            reranked: List[Dict[str, Any]] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                if not isinstance(index, int) or not (0 <= index < len(candidates)):
                    continue
                updated = dict(candidates[index])
                rerank_score = round(float(item.get("relevance_score", updated.get("score", 0.0))), 4)
                if rerank_score < getattr(self, "_rerank_score_threshold", self.DEFAULT_RERANK_SCORE_THRESHOLD):
                    continue
                updated["rerank_score"] = rerank_score
                updated["score"] = rerank_score
                reranked.append(updated)

            if reranked:
                return reranked[:top_n]
        except Exception as ex:
            logger.warning(f"qwen3-rerank 调用失败，回退到 RRF 排序: {ex}")

        return self._apply_rerank_scores(candidates[:top_n])

    def _attach_parent_context(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return []

        parent_collection = self._get_parent_collection()
        if parent_collection is None:
            return items

        parent_ids = [
            item.get("parent_id")
            for item in items
            if isinstance(item, dict) and item.get("parent_id")
        ]
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        if not unique_parent_ids:
            return items

        try:
            parent_results = parent_collection.get(ids=unique_parent_ids)
        except Exception as ex:
            logger.warning(f"回捞父块失败: {ex}")
            return items

        docs = self._flatten_result_rows(parent_results.get("documents", []))
        metas = self._flatten_result_rows(parent_results.get("metadatas", []))
        ids = self._flatten_result_rows(parent_results.get("ids", []))
        parent_map: Dict[str, Dict[str, Any]] = {}

        for parent_id, doc, meta in zip(ids, docs, metas):
            parent_map[str(parent_id)] = {
                "content": doc,
                "meta": meta if isinstance(meta, dict) else {},
            }

        enriched: List[Dict[str, Any]] = []
        for item in items:
            updated = dict(item)
            parent = parent_map.get(str(updated.get("parent_id", "")))
            if parent is not None:
                meta = parent.get("meta", {})
                updated["parent_content"] = parent.get("content", "")
                updated["title"] = meta.get("title", updated.get("title", ""))
                updated["section_title"] = meta.get("section_title", updated.get("section_title", updated.get("title", "")))
                updated["heading_path"] = meta.get("heading_path", updated.get("heading_path", updated.get("title", "")))
            enriched.append(updated)
        return enriched

    def _apply_rerank_scores(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for item in items:
            updated = dict(item)
            if "rerank_score" not in updated:
                updated["rerank_score"] = round(float(updated.get("rrf_score", updated.get("score", 0.0))), 4)
            updated["score"] = updated["rerank_score"]
            ranked.append(updated)
        return ranked

    def _build_candidate_from_meta(
        self,
        meta: Dict[str, Any],
        content: str,
        *,
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
    ) -> Dict[str, Any]:
        child_chunk = meta.get("child_chunk_index", meta.get("chunk_index", 0))
        return {
            "title": meta.get("title", ""),
            "content": content,
            "score": vector_score or bm25_score,
            "vector_score": round(vector_score, 4),
            "bm25_score": round(bm25_score, 4),
            "rrf_score": 0.0,
            "chunk": meta.get("chunk_index", 0),
            "doc_id": meta.get("doc_id", ""),
            "parent_id": meta.get("parent_id", ""),
            "section_title": meta.get("section_title", meta.get("title", "")),
            "heading_path": meta.get("heading_path", meta.get("title", "")),
            "matched_child_content": content,
            "matched_child_chunk": child_chunk,
        }

    def _ensure_bm25_index(self) -> None:
        if getattr(self, "_bm25_doc_tokens", None):
            return
        if not hasattr(self, "_child_records"):
            self._child_records = []
        if not getattr(self, "_child_records", None):
            self._load_child_records_from_collection()
        self._rebuild_bm25_index()

    def _load_child_records_from_collection(self) -> None:
        child_collection = self._get_child_collection()
        if child_collection is None or not hasattr(child_collection, "get"):
            return

        try:
            results = child_collection.get(include=["documents", "metadatas"])
        except TypeError:
            results = child_collection.get()
        except Exception as ex:
            logger.warning(f"加载子块索引失败: {ex}")
            return

        docs = self._flatten_result_rows(results.get("documents", []))
        metas = self._flatten_result_rows(results.get("metadatas", []))
        records: List[Dict[str, Any]] = []
        for doc, meta in zip(docs, metas):
            if not isinstance(meta, dict):
                continue
            records.append({
                "title": meta.get("title", ""),
                "content": doc,
                "doc_id": meta.get("doc_id", ""),
                "parent_id": meta.get("parent_id", ""),
                "section_title": meta.get("section_title", meta.get("title", "")),
                "heading_path": meta.get("heading_path", meta.get("title", "")),
                "chunk_index": meta.get("chunk_index", 0),
                "child_chunk_index": meta.get("child_chunk_index", meta.get("chunk_index", 0)),
                "total_chunks": meta.get("total_chunks", 1),
            })
        self._child_records = records
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        records = getattr(self, "_child_records", [])
        doc_tokens: List[List[str]] = []
        term_freqs: List[Counter[str]] = []
        doc_freq: Dict[str, int] = {}

        total_len = 0
        for record in records:
            tokens = self._tokenize_for_bm25(record.get("content", ""))
            if not tokens:
                tokens = [record.get("content", "")]
            doc_tokens.append(tokens)
            term_freq = Counter(tokens)
            term_freqs.append(term_freq)
            total_len += len(tokens)
            for term in term_freq.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1

        avgdl = (total_len / len(records)) if records else 0.0
        self._bm25_doc_tokens = doc_tokens
        self._bm25_term_freqs = term_freqs
        self._bm25_doc_freq = doc_freq
        self._bm25_avgdl = avgdl

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        text = self._normalize_text(text).lower()
        if not text:
            return []

        tokens: List[str] = []
        for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
            if re.fullmatch(r"[a-z0-9]+", part):
                tokens.append(part)
                continue

            chars = [char for char in part if char.strip()]
            if len(chars) == 1:
                tokens.extend(chars)
                continue

            tokens.append("".join(chars))
            tokens.extend(chars)
            tokens.extend("".join(chars[i:i + 2]) for i in range(len(chars) - 1))

        return tokens

    @staticmethod
    def _build_doc_freq(doc_tokens: List[List[str]]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                freq[term] = freq.get(term, 0) + 1
        return freq

    @staticmethod
    def _candidate_key(item: Dict[str, Any]) -> str:
        parent_id = item.get("parent_id")
        child_chunk = item.get("matched_child_chunk", item.get("child_chunk_index", item.get("chunk", 0)))
        if parent_id:
            return f"{parent_id}:{child_chunk}"

        doc_id = item.get("doc_id")
        if doc_id:
            return f"{doc_id}:{item.get('chunk', 0)}:{child_chunk}"

        content = str(item.get("content", ""))
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return max(0.0, float(raw))
        except ValueError:
            return default

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }

    def _build_doc_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        title = str(doc.get("title", "") or "未命名文档")
        content = self._normalize_text(doc.get("content", ""))
        doc_id = str(doc.get("doc_id") or hashlib.md5(f"{title}:{content[:200]}".encode()).hexdigest())
        return {
            "title": title,
            "doc_id": doc_id,
        }

    def _get_bm25_avgdl(self) -> float:
        return float(getattr(self, "_bm25_avgdl", 0.0))

    def _get_parent_collection(self):
        return getattr(self, "_parent_collection", None)

    def _get_child_collection(self):
        return getattr(self, "_child_collection", getattr(self, "_collection", None))

    def _resolve_search_plan(self, query: str) -> Dict[str, Any]:
        return {"scope": "active", "filters": {}}

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
