import asyncio
import time
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from core.logging_setup import app_log, log_step, log_time
from core.config import COLLECTION_NAME
from models.models import client
from tools.tools import rerank_with_baai, _build_filter, tokenize
from memory.cache import search_cache
from retrieval.bm25_store import bm25_global, all_docs
from tools.tools import encode_query
from utils.utils import _safe_truncate

@log_time
async def search_law(query: str, top_k: int = 15, score_threshold: float = 0.42):
    t0 = time.perf_counter()
    app_log.info(
        "Bắt đầu tìm kiếm",
        extra={"__kv__": {"cau_hoi": _safe_truncate(query, 80), "top_k": top_k, "nguong_diem": score_threshold}},
    )

    cache_key = f"search|{COLLECTION_NAME}|{top_k}|{score_threshold}|{query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("Tìm kiếm từ bộ nhớ cache")
        return cached

    try:
        flt = _build_filter(query)
        has_filter = flt is not None and flt.must

        async def bm25_search_task():
            print(f"DEBUG: Bắt đầu BM25 search task")
            t_bm250 = time.perf_counter()
            if flt:
                print(f"DEBUG: Có filter, fetch filtered docs cho BM25")
                scroll_res, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=flt,
                    limit=1000,
                    with_payload=True,
                )
                filtered_docs = []
                for r in scroll_res:
                    p = r.payload or {}
                    filtered_docs.append({
                        "citation": p.get("exact_citation", ""),
                        "chapter_number": p.get("chapter_number", ""),
                        "article_no": p.get("article_no", ""),
                        "article_title": p.get("article_title", ""),
                        "clause_no": p.get("clause_no", ""),
                        "point_letter": p.get("point_letter", ""),
                        "content": (p.get("content") or "").strip(),
                    })
                tokenized_filtered = [tokenize(d['content']) for d in filtered_docs]
                bm25 = BM25Okapi(tokenized_filtered)
                docs_base = filtered_docs
            else:
                print(f"DEBUG: Không filter, dùng BM25 global")
                bm25 = bm25_global
                docs_base = all_docs

            tokenized_query = tokenize(query)
            bm25_scores = bm25.get_scores(tokenized_query)
            scored_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
            bm25_docs = []
            for idx in scored_indices:
                if bm25_scores[idx] > 0:  # Optional threshold
                    d = docs_base[idx].copy()
                    d['bm25_score'] = float(bm25_scores[idx])
                    bm25_docs.append(d)
            t_bm25 = time.perf_counter() - t_bm250
            print(f"DEBUG: Hoàn tất BM25 search, số docs: {len(bm25_docs)}, thời gian: {t_bm25:.4f}s")
            print(f"DEBUG: Top 3 BM25 docs scores: {[d['bm25_score'] for d in bm25_docs[:3]] if bm25_docs else []}")
            return bm25_docs, t_bm25

        async def embedding_search_task():
            print(f"DEBUG: Bắt đầu embedding search task")
            t_embed0 = time.perf_counter()
            vec = encode_query(query)
            t_embed = time.perf_counter() - t_embed0
            print(f"DEBUG: Hoàn tất encode query, thời gian: {t_embed:.4f}s")

            print(f"DEBUG: Bắt đầu query Qdrant")
            t_q0 = time.perf_counter()
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vec,
                with_payload=True,
                limit=20,
                query_filter=flt,
            )
            t_qdrant = time.perf_counter() - t_q0
            print(f"DEBUG: Hoàn tất embedding search, thời gian: {t_qdrant:.4f}s")

            emb_docs = []
            for r in results.points:
                p = r.payload or {}
                emb_docs.append({
                    "citation": p.get("exact_citation", ""),
                    "chapter_number": p.get("chapter_number", ""),
                    "article_no": p.get("article_no", ""),
                    "article_title": p.get("article_title", ""),
                    "clause_no": p.get("clause_no", ""),
                    "point_letter": p.get("point_letter", ""),
                    "content": (p.get("content") or "").strip(),
                    "embedding_score": float(r.score or 0.0),
                })
            print(f"DEBUG: Số embedding docs: {len(emb_docs)}")
            print(f"DEBUG: Top 3 embedding scores: {[d['embedding_score'] for d in emb_docs[:3]] if emb_docs else []}")
            return emb_docs, t_embed, t_qdrant

        # Chạy song song
        bm25_task = asyncio.create_task(bm25_search_task())
        embedding_task = asyncio.create_task(embedding_search_task())

        bm25_result = await bm25_task
        embedding_result = await embedding_task

        bm25_docs, t_bm25 = bm25_result
        emb_docs, t_embed, t_qdrant = embedding_result

        # Merge and dedup
        print(f"DEBUG: Bắt đầu merge và dedup từ {len(emb_docs)} emb + {len(bm25_docs)} bm25 docs")
        all_unique = {}
        key_func = lambda d: (d.get('article_no', ''), d.get('clause_no', ''), d.get('point_letter', ''))
        for d in emb_docs:
            key = key_func(d)
            if key not in all_unique:
                all_unique[key] = d.copy()
            all_unique[key]['embedding_score'] = d['embedding_score']
            all_unique[key]['bm25_score'] = 0.0  # Default

        for d in bm25_docs:
            key = key_func(d)
            if key not in all_unique:
                all_unique[key] = d.copy()
            all_unique[key]['bm25_score'] = d['bm25_score']
            all_unique[key]['embedding_score'] = all_unique[key].get('embedding_score', 0.0)

        merged_docs = list(all_unique.values())
        print(f"DEBUG: Sau merge, số unique docs: {len(merged_docs)}")

        # Weighted Hybrid Scoring (rerank lần đầu)
        print(f"DEBUG: Bắt đầu weighted hybrid scoring, alpha={0.7 if has_filter else 0.5}, beta={0.3 if has_filter else 0.5}")
        t_rerank0 = time.perf_counter()
        if merged_docs:
            emb_scores = [d['embedding_score'] for d in merged_docs]
            bm25_scores = [d['bm25_score'] for d in merged_docs]
            min_emb, max_emb = min(emb_scores), max(emb_scores) if emb_scores else (0, 0)
            min_bm25, max_bm25 = min(bm25_scores), max(bm25_scores) if bm25_scores else (0, 0)

            for d in merged_docs:
                if max_emb > min_emb:
                    d['norm_emb'] = (d['embedding_score'] - min_emb) / (max_emb - min_emb)
                else:
                    d['norm_emb'] = 0.5 if d['embedding_score'] > 0 else 0.0
                if max_bm25 > min_bm25:
                    d['norm_bm25'] = (d['bm25_score'] - min_bm25) / (max_bm25 - min_bm25)
                else:
                    d['norm_bm25'] = 0.5 if d['bm25_score'] > 0 else 0.0

            # Alpha beta based on query clarity
            if has_filter:
                alpha = 0.7  # embedding high for clear query
                beta = 0.3
            else:
                alpha = 0.5
                beta = 0.5

            for d in merged_docs:
                d['score'] = alpha * d['norm_emb'] + beta * d['norm_bm25']

            ranked = sorted(merged_docs, key=lambda d: d['score'], reverse=True)

            # Clean extra keys
            for d in ranked:
                d.pop('norm_emb', None)
                d.pop('norm_bm25', None)
                d.pop('embedding_score', None)
                d.pop('bm25_score', None)

            # Lấy top 15 sau hybrid rerank và filter threshold
            selected = [d for d in ranked if d['score'] >= score_threshold][:15]
        else:
            selected = []
        t_rerank = time.perf_counter() - t_rerank0
        print(f"DEBUG: Hoàn tất weighted hybrid, số selected docs: {len(selected)}, thời gian: {t_rerank:.4f}s")

        # =========================
        # BAAI Rerank step (rerank lần 2)
        # =========================
        t_baai0 = time.perf_counter()
        if selected:
            print("DEBUG: Bắt đầu rerank bằng BAAI/bge-reranker-base")
            # Rerank chỉ top 15 docs, lấy top 7 tốt nhất
            selected = rerank_with_baai(query, selected, top_k=7)
            print("DEBUG: Hoàn tất rerank bằng BAAI, top1 score:", selected[0].get("baai_score"))
        t_baai = time.perf_counter() - t_baai0
        print(f"DEBUG: Thời gian rerank BAAI: {t_baai:.4f}s")
        # =========================

        search_cache.set(cache_key, selected)

        sk_top1 = selected[0]['score'] if selected and 'score' in selected[0] else (
            selected[0].get("baai_score", 0.0) if selected else 0.0
        )
        log_step(
            "tim_kiem",
            k_yeu_cau=top_k,
            k_tra_ve=len(selected),
            diem_top1=f"{sk_top1:.4f}",
            t_nhung=f"{t_embed:.4f}",
            t_qdrant=f"{t_qdrant:.4f}",
            t_bm25=f"{t_bm25:.4f}",
            t_rerank=f"{t_rerank:.4f}",
            t_baai=f"{t_baai:.4f}",
            t_tong=f"{time.perf_counter()-t0:.4f}",
        )
        app_log.info(
            "Tìm kiếm hoàn tất",
            extra={"__kv__": {"so_luong": len(selected), "diem_top1": f"{sk_top1:.4f}"}})
        return bm25_docs, emb_docs, selected
    except Exception as e:
        app_log.error("Lỗi tìm kiếm", extra={"__kv__": {"loi": str(e)}})
        log_step("tim_kiem_loi", thong_bao=str(e))
        raise
