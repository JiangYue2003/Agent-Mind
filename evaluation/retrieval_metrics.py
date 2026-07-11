import math
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_gold_chunk_keys(path: Path) -> Dict[str, List[str]]:
    """Load the versioned child-chunk gold mapping used for retrieval metrics."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings = payload.get("case_gold_chunk_keys") if isinstance(payload, dict) else None
    if not isinstance(mappings, dict):
        raise ValueError("retrieval gold file must contain case_gold_chunk_keys")
    normalized: Dict[str, List[str]] = {}
    for case_id, keys in mappings.items():
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("retrieval gold case IDs must be non-empty strings")
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys):
            raise ValueError(f"retrieval gold for {case_id!r} must be a non-empty chunk-key list")
        normalized[case_id] = list(dict.fromkeys(keys))
    return normalized


def load_gold_chunk_hashes(path: Path) -> Dict[str, str]:
    """Load SHA-256 fingerprints for the deployed gold child-chunk content."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings = payload.get("chunk_content_sha256") if isinstance(payload, dict) else None
    if not isinstance(mappings, dict):
        raise ValueError("retrieval gold file must contain chunk_content_sha256")
    normalized: Dict[str, str] = {}
    for chunk_key, value in mappings.items():
        digest = str(value).strip().lower()
        if not isinstance(chunk_key, str) or not chunk_key.strip():
            raise ValueError("retrieval gold content hashes must use non-empty chunk keys")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"retrieval gold content hash for {chunk_key!r} must be a SHA-256 hex digest")
        normalized[chunk_key] = digest
    return normalized


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def evaluate_retrieval_records(
    records: List[Dict[str, Any]],
    *,
    gold_chunk_keys: Dict[str, List[str]],
    raw_k: int,
    final_k: int,
) -> Dict[str, Any]:
    """Compute deterministic retrieval metrics for raw and reranked candidates."""
    if raw_k < 1 or final_k < 1:
        raise ValueError("raw_k and final_k must be positive")

    rows: List[Dict[str, Any]] = []
    answerable_rows: List[Dict[str, Any]] = []
    unanswerable_rows: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        if metadata.get("evaluation_mode", "knowledge") != "knowledge":
            continue
        case_id = str(metadata.get("case_id", ""))
        search = record.get("search", {})
        raw_keys = _candidate_keys(search.get("raw_candidates", []), raw_k)
        reranked_keys = _candidate_keys(search.get("reranked_candidates", []), final_k)
        unanswerable = bool(metadata.get("unanswerable", False))
        row: Dict[str, Any] = {
            "case_id": case_id,
            "unanswerable": unanswerable,
            "raw_candidate_count": len(raw_keys),
            "reranked_candidate_count": len(reranked_keys),
        }
        if unanswerable:
            unanswerable_rows.append(row)
        else:
            gold_keys = gold_chunk_keys.get(case_id, [])
            if not gold_keys:
                raise ValueError(f"missing gold_chunk_keys for answerable case {case_id}")
            row["raw"] = _ranking_metrics(raw_keys, gold_keys)
            row["reranked"] = _ranking_metrics(reranked_keys, gold_keys)
            answerable_rows.append(row)
        rows.append(row)

    return {
        "raw_k": raw_k,
        "final_k": final_k,
        "rows": rows,
        "summary": {
            "answerable_cases": len(answerable_rows),
            "unanswerable_cases": len(unanswerable_rows),
            "raw": _average_metrics(answerable_rows, "raw"),
            "reranked": _average_metrics(answerable_rows, "reranked"),
            "unanswerable_noise_rate_raw": _noise_rate(unanswerable_rows, "raw_candidate_count"),
            "unanswerable_noise_rate_reranked": _noise_rate(unanswerable_rows, "reranked_candidate_count"),
        },
    }


def _candidate_keys(candidates: Any, limit: int) -> List[str]:
    if not isinstance(candidates, list):
        return []
    keys: List[str] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = str(candidate.get("chunk_key", "")).strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
        if len(keys) >= limit:
            break
    return keys


def _ranking_metrics(candidate_keys: Iterable[str], gold_keys: Iterable[str]) -> Dict[str, Any]:
    candidates = list(candidate_keys)
    gold = set(gold_keys)
    relevant_ranks = [index for index, key in enumerate(candidates, start=1) if key in gold]
    hit_count = len(relevant_ranks)
    ideal_count = min(len(gold), len(candidates))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "recall_at_k": hit_count / len(gold),
        "hit_at_k": hit_count > 0,
        "mrr_at_k": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "matched_gold_chunk_keys": [key for key in candidates if key in gold],
    }


def _average_metrics(rows: List[Dict[str, Any]], layer: str) -> Dict[str, float]:
    metric_names = ("recall_at_k", "hit_at_k", "mrr_at_k", "ndcg_at_k")
    if not rows:
        return {name: 0.0 for name in metric_names}
    return {
        name: sum(float(row[layer][name]) for row in rows) / len(rows)
        for name in metric_names
    }


def _noise_rate(rows: List[Dict[str, Any]], count_key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[count_key] > 0) / len(rows)
