import asyncio
import time
from typing import List, Dict

from core.logging_setup import app_log, log_step, log_time
from core.config import COLLECTION_NAME
from models.models import client, dense_embedding_model, sparse_embedding_model, late_interaction_embedding_model
from qdrant_client import models
from tools.tools import rerank_with_baai, _build_filter
from memory.cache import search_cache
from utils.utils import _safe_truncate


@log_time
async def search_law(query: str, top_k: int = 10, score_threshold: float = 0.42) -> tuple[List[Dict], List[Dict], List[Dict]]:
    t0 = time.perf_counter()
    app_log.info(
        "Bắt đầu tìm kiếm",
        extra={"__kv__": {"query": _safe_truncate(query, 80), "top_k": top_k}},
    )

    cache_key = f"search|{COLLECTION_NAME}|{top_k}|{score_threshold}|{query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("Tìm trong cache ✅")
        return [], [], cached

    try:
        flt = _build_filter(query)

        print("DEBUG: → Bắt đầu hybrid (dense + sparse) + ColBERT Rerank")
        t_hybrid0 = time.perf_counter()

        # 1️⃣ Tạo embedding query
        dense_vectors = next(dense_embedding_model.query_embed(query))
        sparse_vectors = next(sparse_embedding_model.query_embed(query))
        late_vectors = next(late_interaction_embedding_model.query_embed(query))  # ColBERT

        # 2️⃣ Prefetch (dense + sparse)
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="bge-m3",
                limit=50,
                filter=flt,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vectors.indices,
                    values=sparse_vectors.values
                ),
                using="bm25",
                limit=50,
                filter=flt,
            )
        ]

        # 3️⃣ Hybrid + ColBERT rerank → top ~20 ứng viên
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=late_vectors,
            using="ColBERT-v2",
            query_filter=flt,
            with_payload=True,
            limit=20,
        )

        colbert_docs = []
        for point in results.points:
            payload = point.payload or {}
            colbert_docs.append({
                "chapter_number": payload.get("chapter_number", ""),
                "article_no": payload.get("article_no", ""),
                "article_title": payload.get("article_title", ""),
                "clause_no": payload.get("clause_no", ""),
                "point_letter": payload.get("point_letter", ""),
                "content": (payload.get("content") or "").strip(),
                "colbert_score": point.score,
            })

        print(f"DEBUG: Hybrid + ColBERT done ✅ | count: {len(colbert_docs)}")

        # 4️⃣ BAAI rerank top 20 → top 10
        t_baai0 = time.perf_counter()
        if colbert_docs:
            selected = rerank_with_baai(query, colbert_docs, top_k=top_k)
            selected = [doc for doc in selected if doc.get("baai_score", 0.0) >= score_threshold]
        else:
            selected = []

        print(f"DEBUG: BAAI rerank done ✅ | final: {len(selected)}")

        t_hybrid = time.perf_counter() - t_hybrid0
        t_baai = time.perf_counter() - t_baai0

        search_cache.set(cache_key, selected)

        sk_top1 = selected[0].get("baai_score", 0.0) if selected else 0.0
        log_step(
            "hybrid_search",
            k_tra_ve=len(selected),
            top1=f"{sk_top1:.4f}",
            t_hybrid=f"{t_hybrid:.4f}",
            t_baai=f"{t_baai:.4f}",
        )
        return [], [], selected

    except Exception as e:
        app_log.error("Lỗi tìm kiếm ❌", extra={"__kv__": {"error": str(e)}})
        log_step("tim_kiem_loi", error=str(e))
        raise
