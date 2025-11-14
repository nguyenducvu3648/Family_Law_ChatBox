import os
import re
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from qdrant_client.http.models import Filter as QFilter, FieldCondition, MatchValue
load_dotenv()

# =========================
# CONFIG
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "BAAI_BDS_HYBRID"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CASUAL_MAX_WORDS = int(os.getenv("CASUAL_MAX_WORDS", "0").strip() or 0)
FALLBACK_CASUAL = os.getenv(
    "FALLBACK_CASUAL",
    "Chào bạn, mình có thể hỗ trợ câu hỏi về Luật Bất động sản. Bạn muốn hỏi nội dung gì?",
).strip()

# =========================
# LOGGING SETUP
# =========================
class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        extras_dict = getattr(record, "__kv__", {})
        if isinstance(extras_dict, dict):
            for k, v in extras_dict.items():
                try:
                    extras.append(f"{k}={v}")
                except Exception:
                    continue
        return base + (" | " + ",".join(extras) if extras else "")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
handlers = [logging.StreamHandler()]

try:
    fh = logging.FileHandler("realestate_assistant.log", encoding="utf-8")
    handlers.append(fh)
except Exception:
    pass

for h in handlers:
    h.setFormatter(KVFormatter(LOG_FORMAT))

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
for h in handlers:
    root_logger.addHandler(h)

app_log = logging.getLogger("app")
metrics_logger = logging.getLogger("metrics")

if not metrics_logger.handlers:
    try:
        fhm = logging.FileHandler("metrics.log", encoding="utf-8")
        fhm.setFormatter(logging.Formatter("%(message)s"))
        metrics_logger.addHandler(fhm)
        metrics_logger.setLevel(logging.INFO)
    except Exception:
        pass

def log_step(event: str, **kv):
    kvpairs = ",".join([f"{k}={v}" for k, v in kv.items()])
    try:
        metrics_logger.info(f"ts={int(time.time())},evt={event},{kvpairs}")
    except Exception:
        pass

def log_time(func):
    import functools
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            app_log.info(
                f"Thời gian thực thi {func.__name__}",
                extra={"__kv__": {"thoi_gian": f"{elapsed:.4f} giây"}},
            )
    return sync_wrapper

# =========================
# UTILS
# =========================
LEGAL_HINTS = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|đất|nhà ở|đầu tư|xây dựng|kinh doanh bất động sản|thuế|sổ đỏ|sổ hồng|quy hoạch|cho thuê|mua bán|chuyển nhượng)\b"
)

def looks_like_legal(query: str) -> bool:
    return bool(LEGAL_HINTS.search(query or ""))

def _safe_truncate(text: str, limit: int = 800) -> str:
    return text if text and len(text) <= limit else (text[:limit] + "…(cắt)") if text else ""

def normalize_legal_query(query: str) -> str:
    """Chuẩn hóa câu hỏi: sửa chính tả, thêm dấu hỏi, loại bỏ khoảng trắng thừa"""
    original = (query or "").strip()
    text = re.sub(r"\s+", " ", original).strip()
    text = text[0].upper() + text[1:] if text else text

    corrections = {
        "sổ đỏd": "sổ đỏ",
        "sổ hông": "sổ hồng",
        "thuế sd đất": "thuế sử dụng đất",
        "kinh doanh bds": "kinh doanh bất động sản",
        "pháp luât": "pháp luật",
        "đươc": "được",
    }
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)

    text = re.sub(r"\s*[.!]+\s*", ". ", text)
    text = text.strip().rstrip(".").strip()
    
    # Thêm dấu hỏi nếu chưa có
    if not text.endswith("?"):
        text += "?"
    
    text = re.sub(r"[!?]{2,}", "?", text)
    text = text.replace(",,", ",").replace("..", ".")
    return text.strip()

# =========================
# CACHE
# =========================
class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 1800, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max = max_items
        self.store: Dict[str, Tuple[float, Any]] = {}

    def _evict_if_needed(self):
        if len(self.store) <= self.max:
            return
        oldest_key = min(self.store, key=lambda k: self.store[k][0])
        self.store.pop(oldest_key, None)

    def get(self, key: str):
        item = self.store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        self.store[key] = (time.time(), value)
        self._evict_if_needed()

embed_cache = SimpleTTLCache(ttl_seconds=3600, max_items=1024)
search_cache = SimpleTTLCache(ttl_seconds=900, max_items=1024)

# =========================
# PROMPTS
# =========================
QUERY_ROUTING_PROMPT = r'''
Bạn là một hệ thống định tuyến truy vấn chuyên nghiệp cho tra cứu LUẬT BẤT ĐỘNG SẢN Việt Nam.
Nhiệm vụ: Phân tích câu hỏi và quyết định phương thức truy vấn tối ưu.

=== CÁC HÀNH ĐỘNG (query_action) ===

1. "casual": 
   - Câu hỏi xã giao, chào hỏi, cảm ơn
   - VÍ DỤ: "Chào bạn", "Cảm ơn", "Bạn khỏe không"
   - Trả lời ngắn gọn trong "casual_answer"

2. "fetch":
   - Câu hỏi NHẮC CỤ THỂ SỐ ĐIỀU/KHOẢN/ĐIỂM/CHƯƠNG
   - VÍ DỤ: "Điều 10 Luật Đất đai quy định gì", "Khoản 2 Điều 15"
   - Điền đầy đủ "filters" và tạo "search_query" mô tả nội dung cần tìm

3. "rag_search":
   - Câu hỏi pháp lý TỔNG QUÁT, cần giải thích/so sánh
   - VÍ DỤ: "Phân biệt sổ đỏ và sổ hồng", "Quy trình chuyển nhượng đất"
   - Tinh chỉnh câu hỏi thành "search_query" rõ ràng, chi tiết

4. "hybrid":
   - Câu hỏi VỪA có điều khoản cụ thể VỪA cần ngữ cảnh rộng
   - VÍ DỤ: "Điều 10 áp dụng như thế nào trong trường hợp tranh chấp"
   - Điền CẢ "filters" VÀ "search_query"

=== QUY TẮC TRÍCH XUẤT FILTERS ===
- article_no: Số nguyên (10, 15, 127...)
- clause_no: Số nguyên (1, 2, 3...)
- point_letter: Chữ cái thường (a, b, c...)
- chapter_number: Số nguyên (1, 2, 3...)

=== YÊU CẦU ===
- Trả về JSON thuần (KHÔNG markdown, KHÔNG ```json)
- search_query: LUÔN là câu hỏi đã được mở rộng, chi tiết hóa
- QUAN TRỌNG: Chỉ điền "filters" khi câu hỏi CÓ CHỨA SỐ CỤ THỂ (ví dụ: "Điều 5", "Khoản 1").
- Nếu câu hỏi chung chung (ví dụ: "Quy trình bán đất", "Thủ tục làm sổ đỏ"), hãy để "filters": {} (RỖNG).

=== SCHEMA ===
{
  "query_action": "casual|fetch|rag_search|hybrid",
  "search_query": "câu hỏi đã tinh chỉnh, mở rộng chi tiết",
  "filters": {
    "article_no": int hoặc null,
    "clause_no": int hoặc null,
    "point_letter": "a" hoặc null,
    "chapter_number": int hoặc null
  },
  "casual_answer": "câu trả lời ngắn (chỉ khi query_action=casual)"
}
'''.strip()

ANSWER_PROMPT = r'''
Bạn là chuyên gia pháp lý về LUẬT BẤT ĐỘNG SẢN Việt Nam.
Nhiệm vụ: Giải thích điều luật một cách dễ hiểu, chính xác dựa trên ngữ cảnh được cung cấp.

YÊU CẦU:
- Trích dẫn chính xác điều/khoản/điểm
- Giải thích bằng ngôn ngữ đời thường
- Nêu ví dụ thực tế nếu có thể
- Cảnh báo các trường hợp ngoại lệ
'''.strip()

# =========================
# MODELS & EXTERNAL CLIENTS
# =========================
genai.configure(api_key=GEMINI_API_KEY)
routing_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=QUERY_ROUTING_PROMPT)
answer_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=ANSWER_PROMPT)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

# =========================
# Embedding models
# =========================


# Dense embedding: SentenceTransformer + BGE-M3 multilingual 1024 dim
dense_embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Sparse embedding (BM25) và Late Interaction (ColBERT) giữ nguyên
sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

# Reranker giữ nguyên
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rerank_model.eval()

# =========================
# RERANK & TOOLS
# =========================
def rerank_with_baai(query, docs, top_k=15):
    if not docs:
        return docs
    try:
        pairs = [(query, d["content"]) for d in docs]
        inputs = rerank_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        with torch.no_grad():
            scores = rerank_model(**inputs).logits.view(-1).float()
        app_log.info(f"BAAI scores: {scores.tolist()}")  # Debug scores
        for d, s in zip(docs, scores):
            d["baai_score"] = float(s)
        reranked = sorted(docs, key=lambda x: x["baai_score"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        app_log.warning(f"Rerank lỗi: {e}")
        for d in docs:
            d["baai_score"] = float(d.get("colbert_score") or d.get("score") or 0.0)
        return sorted(docs, key=lambda x: x["baai_score"], reverse=True)[:top_k]

def _build_filter(filters: Dict[str, Any]):
    """Xây dựng QFilter từ dict filters"""
    must = []
    mapping = {
        "article_no": ("metadata.article_no", int),
        "clause_no": ("metadata.clause_no", int),
        "point_letter": ("metadata.point_letter", str),
        "chapter_number": ("metadata.chapter_number", int)
    }
    for key, (field_path, caster) in mapping.items():
        if key in filters and filters[key] not in (None, "", 0):
            try:
                val = caster(filters[key])
                must.append(FieldCondition(key=field_path, match=MatchValue(value=val)))
            except Exception as e:
                app_log.warning(f"Lỗi ép kiểu filter {key}: {e}")
    return QFilter(must=must) if must else None

# =========================
# FETCH
# =========================
@log_time
def _fetch(filters: Dict[str, Any], limit: int = 10) -> List[Dict]:
    """Tìm kiếm theo filters cụ thể (Điều/Khoản/Điểm/Chương)"""
    flt = _build_filter(filters)
    if not flt:
        app_log.warning("Không có điều kiện filter hợp lệ")
        return []
    
    out = []
    try:
        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=min(64, max(5, limit)),
            with_payload=True
        )
        app_log.info(f"Fetch: {len(scroll_res)} docs", extra={"__kv__": {"filters": str(filters)}})
        
        for r in scroll_res:
            p = getattr(r, "payload", {}) or {}
            meta = p.get("metadata", {})
            out.append({
                "chapter": meta.get("chapter", ""),
                "chapter_number": meta.get("chapter_number", ""),
                "article_no": meta.get("article_no", ""),
                "article_title": meta.get("article_title", ""),
                "clause_no": meta.get("clause_no", ""),
                "point_letter": meta.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": 1.0,
            })
            if len(out) >= limit:
                break
    except Exception as e:
        app_log.error(f"Lỗi fetch: {e}")
        return []
    return out

# =========================
# RAG SEARCH
# =========================

# ===== OLD FLOW (COMMENTED OUT - Collection lỗi) =====
# @log_time
# async def search_law_old(original_query: str, normalized_query: str, top_k: int = 10):
#     """Hybrid search: Dense + Sparse + ColBERT + BAAI rerank với Multi-Query Ensemble"""
#     app_log.info(f"RAG search ensemble: original={_safe_truncate(original_query, 80)}, normalized={_safe_truncate(normalized_query, 80)}")
#     
#     cache_key = f"search|{COLLECTION_NAME}|{top_k}|{original_query}|{normalized_query}"
#     cached = search_cache.get(cache_key)
#     if cached is not None:
#         app_log.info("Cache hit ✅")
#         return [], [], cached
#     
#     try:
#         # Hàm helper để chạy hybrid search cho một query
#         async def hybrid_for_query(q: str) -> List[Dict]:
#             # Embeddings
#             dense_vectors = dense_embedding_model.encode(q).tolist()  # Ensure list[float]
#             sparse_vectors = next(sparse_embedding_model.query_embed(q))
#             late_vectors = next(late_interaction_embedding_model.query_embed(q))
#             
#             # Hybrid search với ColBERT
#             from qdrant_client import models
#             prefetch = [
#                 models.Prefetch(query=dense_vectors, using="bge-m3", limit=20),
#                 models.Prefetch(
#                     query=models.SparseVector(
#                         indices=sparse_vectors.indices.tolist(),
#                         values=sparse_vectors.values.tolist(),
#                     ),
#                     using="bm25",
#                     limit=10
#                 )
#             ]
#             
#             results = client.query_points(
#                 collection_name=COLLECTION_NAME,
#                 prefetch=prefetch,
#                 query=late_vectors,
#                 using="colbertv2.0",
#                 with_payload=True,
#                 limit=10
#             )
#             
#             # Parse results
#             points = getattr(results, "points", [])
#             app_log.info(f"Points retrieved for '{q[:20]}...': {len(points)}")
#             colbert_docs = []
#             for point in points:
#                 payload = getattr(point, "payload", {}) or {}
#                 meta = payload.get("metadata", {})
#                 content = (payload.get("content") or "").strip()
#                 colbert_docs.append({
#                     "id": getattr(point, "id", ""),
#                     "chapter_number": meta.get("chapter_number", ""),
#                     "chapter": meta.get("chapter", ""),
#                     "article_no": meta.get("article_no", ""),
#                     "article_title": meta.get("article_title", ""),
#                     "clause_no": meta.get("clause_no", ""),
#                     "point_letter": meta.get("point_letter", ""),
#                     "content": content,
#                     "colbert_score": getattr(point, "score", 0.0),
#                 })
#                 app_log.debug(f"Doc content: {content[:100]}...")
#             return colbert_docs
#         
#         # Chạy song song 2 hybrid searches
#         docs_a_task = asyncio.create_task(hybrid_for_query(original_query))
#         docs_b_task = asyncio.create_task(hybrid_for_query(normalized_query))
#         docs_a = await docs_a_task
#         docs_b = await docs_b_task
#         
#         # Merge: Gộp và loại trùng dựa trên id
#         merged_docs = {}
#         for doc in docs_a + docs_b:
#             doc_id = doc.get("id")
#             if doc_id and doc_id not in merged_docs:
#                 merged_docs[doc_id] = doc
#                 app_log.info(f"Merged doc ID {doc_id}: content {doc['content'][:100]}... score {doc['colbert_score']}")
#         merged_list = list(merged_docs.values())
#         
#         # Chỉ lấy top_k docs theo colbert_score
#         selected = sorted(merged_list, key=lambda x: x.get("colbert_score", 0.0), reverse=True)[:top_k]
#         
#         search_cache.set(cache_key, selected)
#         app_log.info(f"RAG ensemble done: {len(selected)} docs (A={len(docs_a)}, B={len(docs_b)}, Merged={len(merged_list)})")
#         
#         return [], [], selected
#     except Exception as e:
#         app_log.error(f"RAG search ensemble lỗi: {e}", exc_info=True)
#         return [], [], []

# ===== NEW FLOW: Dense + Sparse + RRF =====
def reciprocal_rank_fusion(dense_docs: List[Dict], sparse_docs: List[Dict], k: int = 60, top_k: int = 10) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) để merge kết quả từ Dense và Sparse search
    Formula: RRF_score(doc) = sum(1 / (k + rank_i)) for all rankings
    """
    rrf_scores = {}
    doc_map = {}
    
    # Score từ Dense search
    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = doc.get("id")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            doc_map[doc_id] = doc
    
    # Score từ Sparse search
    for rank, doc in enumerate(sparse_docs, start=1):
        doc_id = doc.get("id")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
    
    # Sắp xếp theo RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Tạo danh sách kết quả
    results = []
    for doc_id in sorted_ids[:top_k]:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)
    
    app_log.info(f"RRF fusion: {len(dense_docs)} dense + {len(sparse_docs)} sparse → {len(results)} merged")
    return results

@log_time
async def search_law(original_query: str, normalized_query: str, top_k: int = 10):
    """
    NEW FLOW: Dense (20) + Sparse (10) search song song, sau đó RRF fusion → top_k
    CHỈ chạy với original_query (normalized_query bị comment)
    """
    app_log.info(f"RAG search (Dense+Sparse+RRF): original={_safe_truncate(original_query, 80)}")
    
    cache_key = f"search_v2|{COLLECTION_NAME}|{top_k}|{original_query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("Cache hit ✅")
        return [], [], cached
    
    try:
        # Helper function: Dense + Sparse search cho 1 query
        async def hybrid_search_for_query(q: str) -> tuple[List[Dict], List[Dict]]:
            """Trả về (dense_docs, sparse_docs)"""
            # 1. Dense search (top 20)
            dense_vectors = dense_embedding_model.encode(q).tolist()
            from qdrant_client import models
            
            dense_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vectors,
                using="bge-m3",
                limit=20,
                with_payload=True
            )
            
            dense_docs = []
            points = getattr(dense_results, "points", [])
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                meta = payload.get("metadata", {})
                dense_docs.append({
                    "id": getattr(point, "id", ""),
                    "chapter_number": meta.get("chapter_number", ""),
                    "chapter": meta.get("chapter", ""),
                    "article_no": meta.get("article_no", ""),
                    "article_title": meta.get("article_title", ""),
                    "clause_no": meta.get("clause_no", ""),
                    "point_letter": meta.get("point_letter", ""),
                    "content": (payload.get("content") or "").strip(),
                    "dense_score": getattr(point, "score", 0.0),
                })
            
            # 2. Sparse search (top 10)
            sparse_vectors = next(sparse_embedding_model.query_embed(q))
            
            sparse_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=models.SparseVector(
                    indices=sparse_vectors.indices.tolist(),
                    values=sparse_vectors.values.tolist(),
                ),
                using="bm25",
                limit=10,
                with_payload=True
            )
            
            sparse_docs = []
            points = getattr(sparse_results, "points", [])
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                meta = payload.get("metadata", {})
                sparse_docs.append({
                    "id": getattr(point, "id", ""),
                    "chapter_number": meta.get("chapter_number", ""),
                    "chapter": meta.get("chapter", ""),
                    "article_no": meta.get("article_no", ""),
                    "article_title": meta.get("article_title", ""),
                    "clause_no": meta.get("clause_no", ""),
                    "point_letter": meta.get("point_letter", ""),
                    "content": (payload.get("content") or "").strip(),
                    "sparse_score": getattr(point, "score", 0.0),
                })
            
            app_log.info(f"Query '{q[:30]}...': Dense={len(dense_docs)}, Sparse={len(sparse_docs)}")
            return dense_docs, sparse_docs
        
        # CHỈ chạy cho original_query (COMMENT normalized_query)
        dense_docs, sparse_docs = await hybrid_search_for_query(original_query)
        
        # ===== COMMENTED: Multi-query ensemble với normalized_query =====
        # task_a = asyncio.create_task(hybrid_search_for_query(original_query))
        # task_b = asyncio.create_task(hybrid_search_for_query(normalized_query))
        # 
        # (dense_a, sparse_a) = await task_a
        # (dense_b, sparse_b) = await task_b
        # 
        # # Merge kết quả từ 2 queries (loại trùng)
        # all_dense = {}
        # for doc in dense_a + dense_b:
        #     doc_id = doc.get("id")
        #     if doc_id and doc_id not in all_dense:
        #         all_dense[doc_id] = doc
        # 
        # all_sparse = {}
        # for doc in sparse_a + sparse_b:
        #     doc_id = doc.get("id")
        #     if doc_id and doc_id not in all_sparse:
        #         all_sparse[doc_id] = doc
        # 
        # # RRF fusion
        # dense_list = list(all_dense.values())
        # sparse_list = list(all_sparse.values())
        
        # Sắp xếp theo score trước khi RRF
        dense_list = sorted(dense_docs, key=lambda x: x.get("dense_score", 0.0), reverse=True)
        sparse_list = sorted(sparse_docs, key=lambda x: x.get("sparse_score", 0.0), reverse=True)
        
        selected = reciprocal_rank_fusion(dense_list, sparse_list, k=60, top_k=top_k)
        
        search_cache.set(cache_key, selected)
        app_log.info(f"RRF done: {len(selected)} final docs")
        
        # Return format: (bm25_docs, emb_docs, final_docs)
        return sparse_list[:5], dense_list[:5], selected
        
    except Exception as e:
        app_log.error(f"RAG search lỗi: {e}", exc_info=True)
        return [], [], []
# =========================
# QUERY ROUTING (thay thế Intent Analysis)
# =========================
@log_time
def route_query(query: str) -> Dict[str, Any]:
    """
    Định tuyến truy vấn: quyết định phương thức xử lý (casual/fetch/rag_search/hybrid)
    Thay thế hoàn toàn analyze_intent()
    """
    # Bước 1: Chuẩn hóa câu hỏi
    normalized = normalize_legal_query(query)
    
    try:
        # Bước 2: Gọi LLM routing
        cfg = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
        )
        
        resp = routing_model.generate_content(
            f"Câu hỏi: {normalized}",
            generation_config=cfg,
        )
        
        # Parse response
        raw = ""
        try:
            if hasattr(resp, "candidates") and resp.candidates:
                parts = getattr(resp.candidates[0].content, "parts", [])
                if parts and hasattr(parts[0], "text"):
                    raw = parts[0].text
        except Exception as e:
            app_log.warning(f"Không đọc được response: {e}")
        
        if not raw:
            # Fallback: heuristics
            if looks_like_legal(normalized):
                return {
                    "query_action": "rag_search",
                    "search_query": normalized,
                    "filters": {},
                    "casual_answer": ""
                }
            else:
                return {
                    "query_action": "casual",
                    "search_query": "",
                    "filters": {},
                    "casual_answer": FALLBACK_CASUAL
                }
        
        # Parse JSON
        data = json.loads(raw) if raw else {}
        
        # Validate và chuẩn hóa
        query_action = data.get("query_action", "rag_search")
        if query_action not in {"casual", "fetch", "rag_search", "hybrid"}:
            query_action = "rag_search" if looks_like_legal(normalized) else "casual"
        
        result = {
            "query_action": query_action,
            "search_query": data.get("search_query", normalized),
            "filters": data.get("filters", {}),
            "casual_answer": data.get("casual_answer", "")
        }
        
        app_log.info(
            f"Routing: {query_action}",
            extra={"__kv__": {
                "action": query_action,
                "has_filters": bool(result["filters"]),
                "query_len": len(result["search_query"])
            }}
        )
        
        return result
        
    except Exception as e:
        app_log.error(f"Routing lỗi: {e}")
        # Fallback
        if looks_like_legal(normalized):
            return {
                "query_action": "rag_search",
                "search_query": normalized,
                "filters": {},
                "casual_answer": ""
            }
        else:
            return {
                "query_action": "casual",
                "search_query": "",
                "filters": {},
                "casual_answer": FALLBACK_CASUAL
            }

# =========================
# QUERY REWRITER
# =========================
REWRITE_INSTRUCTIONS = """
You are an expert at reformulating legal questions to be more precise and detailed.
Expand acronyms, add context, make it search-friendly.
Return ONLY the rewritten query without any additional text.
"""

@log_time
def rewrite_query(query: str) -> str:
    """Tinh chỉnh câu hỏi để tối ưu RAG search"""
    if not (query and query.strip()):
        return query
    try:
        cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=96)
        prompt = f"{REWRITE_INSTRUCTIONS}\n\nUser question: {query}\nOutput:"
        
        resp = routing_model.generate_content(prompt, generation_config=cfg)
        text = (getattr(resp, "text", None) or "").strip()
        
        if not text and getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            if parts and hasattr(parts[0], "text"):
                text = parts[0].text
        
        out = text.splitlines()[0].strip() if text else query
        app_log.info(f"Query rewritten: {query[:60]} → {out[:60]}")
        return out or query
    except genai.types.BlockedPromptException as be:
        app_log.warning(f"Gemini blocked (reason=2?): {be}. Fallback to original query.")
        return query
    except Exception as e:
        app_log.warning(f"Rewrite lỗi: {e}. Fallback to original query.")
        return query

# =========================
# RENDER
# =========================
def docs_to_markdown(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "(Chưa có dữ liệu)"
    lines = []
    for i, doc in enumerate(docs, 1):
        article_no = doc.get("article_no", "")
        article_title = doc.get("article_title", "")
        clause_no = doc.get("clause_no", "")
        point_letter = doc.get("point_letter", "")
        content = doc.get("content", "")[:150]
        # Lấy score phù hợp với loại search
        score = doc.get("rrf_score") or doc.get("dense_score") or doc.get("sparse_score") or doc.get("baai_score") or doc.get("colbert_score") or doc.get("score", 0.0)
        
        citation_parts = []
        if article_no:
            citation_parts.append(f"Điều {article_no}")
        if clause_no:
            citation_parts.append(f"Khoản {clause_no}")
        if point_letter:
            citation_parts.append(f"Điểm {point_letter}")
        citation = " ".join(citation_parts) if citation_parts else "N/A"
        
        lines.append(f"**{i}. {citation}**")
        if article_title:
            lines.append(f"*{article_title}*")
        lines.append(f"Score: {score:.4f}")
        lines.append(f"{content}...")
        lines.append("")
    return "\n".join(lines)

def paginate_docs(docs: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[List[Dict], int, int, int]:
    if not docs:
        return [], 0, 0, 1
    total = len(docs)
    total_pages = (total + page_size - 1) // page_size
    current_page = max(1, min(page, total_pages))
    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = docs[start_idx:end_idx]
    return paginated, total, total_pages, current_page

def docs_page_markdown(docs: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[str, str]:
    if not docs:
        return "(Chưa có dữ liệu)", " Trang 0/0"
    
    paginated, total, total_pages, current_page = paginate_docs(docs, page, page_size)
    lines = []
    start_idx = (current_page - 1) * page_size
    
    for i, doc in enumerate(paginated, start=start_idx + 1):
        article_no = doc.get("article_no", "")
        article_title = doc.get("article_title", "")
        clause_no = doc.get("clause_no", "")
        point_letter = doc.get("point_letter", "")
        content = doc.get("content", "")
        # Lấy score phù hợp với loại search
        score = doc.get("rrf_score") or doc.get("dense_score") or doc.get("sparse_score") or doc.get("baai_score") or doc.get("colbert_score") or doc.get("score", 0.0)
        
        citation_parts = []
        if article_no:
            citation_parts.append(f"Điều {article_no}")
        if clause_no:
            citation_parts.append(f"Khoản {clause_no}")
        if point_letter:
            citation_parts.append(f"Điểm {point_letter}")
        citation = " ".join(citation_parts) if citation_parts else "N/A"
        
        lines.append(f"### {i}. {citation}")
        if article_title:
            lines.append(f"**{article_title}**")
        lines.append(f"*Score: {score:.4f}*")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")
    
    markdown = "\n".join(lines)
    page_label = f" Trang {current_page}/{total_pages}"
    return markdown, page_label

# =========================
# PROMPT BUILD
# =========================
def build_prompt(query: str, docs: List[Dict[str, Any]], history_msgs=None):
    history_block = ""
    if history_msgs:
        lines = []
        for i, m in enumerate(history_msgs[-5:], 1):
            role = m.get("role", "")
            content = m.get("content", "")
            role_label = "Người dùng" if role == "user" else "Trợ lý"
            lines.append(f"- {i}. {role_label}: {content}")
        history_block = "\nLịch sử hội thoại gần đây:\n" + "\n".join(lines)
    
    # Sắp xếp docs theo độ ưu tiên
    def doc_priority(d):
        text = (d.get("content") or "").lower()
        if re.search(r"\b(trừ khi|ngoại lệ|nếu chưa bị tuyên bố|trường hợp|nhưng không)\b", text):
            return 0
        return 1
    
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            doc_priority(d),
            int(d.get("article_no") or 9999),
            int(d.get("clause_no") or 9999),
            str(d.get("point_letter") or ""),
        ),
    )
    
    context_lines = []
    citations = []
    for idx, d in enumerate(docs_sorted, 1):
        art = d.get("article_no")
        cls = d.get("clause_no")
        pt = d.get("point_letter")
        parts = []
        if pt:
            parts.append(f"Điểm {pt}")
        if cls:
            parts.append(f"Khoản {cls}")
        if art:
            parts.append(f"Điều {art}")
        cited = " ".join(parts)
        chapter = f" (Chương {d.get('chapter_number')})" if d.get("chapter_number") else ""
        title = f" — {d.get('article_title')}" if d.get("article_title") else ""
        content = (d.get("content") or "").strip()
        context_lines.append(f"{idx}) {cited}{chapter}{title}: {content}")
        citations.append(f"- {cited}{title}: {content[:300]}")
    
    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."
    citations_block = "\n".join(citations) if citations else "❌ Không có điều luật nào."
    
    prompt = f"""
{history_block}

Câu hỏi người dùng: {query}

Ngữ cảnh pháp lý:
{context}

Hãy trả lời câu hỏi dựa trên ngữ cảnh trên. Trích dẫn chính xác điều/khoản/điểm và giải thích dễ hiểu.
"""
    return prompt

# =========================
# LLM STREAMING
# =========================
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gemini_stream(prompt, temperature: float):
    cfg = genai.types.GenerationConfig(temperature=float(temperature))
    return answer_model.generate_content(prompt, generation_config=cfg, stream=True)

@log_time
def stream_answer(prompt, temperature=0.2):
    t0 = time.perf_counter()
    try:
        resp = _gemini_stream(prompt, temperature)
        for ch in resp:
            if getattr(ch, "text", None):
                yield ch.text
    except Exception as e:
        app_log.error(f"LLM streaming lỗi: {e}")
        yield f"\n\nLỗi gọi mô hình: {e}"
    finally:
        log_step("llm_tong", thoi_gian=f"{time.perf_counter()-t0:.4f}")

# =========================
# UI HELPER
# =========================
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    GRADIO_AVAILABLE = False
    app_log.warning("gradio không khả dụng")

CSS = """
#chatbot { height: 540px !important; }
label { font-size:12px !important; opacity:.9 }
#cites-box {
    max-height: 360px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
#bm25-box, #emb-box {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
"""

def ui_return(msg_val, chatbot_val, bm25_val, emb_val, cites_val, last_answer_val, docs_val, page_val, page_label_val, history_msgs):
    if GRADIO_AVAILABLE:
        return (
            msg_val,
            chatbot_val,
            gr.update(value=bm25_val),
            gr.update(value=emb_val),
            gr.update(value=cites_val),
            last_answer_val,
            docs_val,
            page_val,
            page_label_val,
            history_msgs,
        )
    else:
        return (
            msg_val,
            chatbot_val,
            bm25_val,
            emb_val,
            cites_val,
            last_answer_val,
            docs_val,
            page_val,
            page_label_val,
            history_msgs,
        )

# =========================
# MAIN RESPONSE GENERATOR (REFACTORED - NO INTENT)
# =========================
@log_time
def respond_generator(message, history_msgs, cur_page_size, k=15, temperature=0.2):
    """
    Luồng xử lý chính - ĐÃ LOẠI BỎ INTENT ANALYSIS
    """
    print(f"DEBUG: Xử lý câu hỏi: {message}")
    
    if not (message and message.strip()):
        print("DEBUG: Câu hỏi rỗng")
        if GRADIO_AVAILABLE:
            gr.Info("Vui lòng nhập câu hỏi.")
        yield ui_return(
            gr.update(value="") if GRADIO_AVAILABLE else "",
            history_msgs,
            "",
            "",
            "",
            "",
            [],
            1,
            " Trang 0/0",
            history_msgs,
        )
        return
    
    try:
        # ===== BƯỚC 1: ROUTING (thay thế Intent Analysis) =====
        routing = route_query(message)
        query_action = routing["query_action"]
        search_query = routing["search_query"]
        filters = routing["filters"]
        casual_answer = routing["casual_answer"]
        
        log_step("routing", action=query_action, has_filters=bool(filters))
        
        # ===== BƯỚC 2: XỬ LÝ CASUAL =====
        if query_action == "casual":
            final_answer = casual_answer.strip() if casual_answer else FALLBACK_CASUAL
            
            # Cắt ngắn nếu cần
            if final_answer and CASUAL_MAX_WORDS > 0:
                words = final_answer.split()
                if len(words) > CASUAL_MAX_WORDS:
                    final_answer = " ".join(words[:CASUAL_MAX_WORDS])
            
            if len(final_answer) >= 1:
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": final_answer},
                ]
                yield ui_return(
                    gr.update(value="") if GRADIO_AVAILABLE else "",
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    final_answer,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
                return
            
            # Fallback: dùng LLM sinh câu trả lời ngắn
            simple_prompt = f"Trả lời thân thiện ngắn gọn (<=2 câu) tiếng Việt cho câu: {message}"
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            
            acc = ""
            yield ui_return(
                gr.update(value="") if GRADIO_AVAILABLE else "",
                history_msgs,
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                acc,
                [],
                1,
                " Trang 0/0",
                history_msgs,
            )
            
            buffer = ""
            for chunk in stream_answer(simple_prompt, temperature=float(temperature)):
                buffer += chunk
                if len(buffer) >= 50:
                    acc += buffer
                    history_msgs[-1]["content"] = acc
                    yield ui_return(
                        gr.update(value="") if GRADIO_AVAILABLE else "",
                        history_msgs,
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        acc,
                        [],
                        1,
                        " Trang 0/0",
                        history_msgs,
                    )
                    buffer = ""
            if buffer:
                acc += buffer
                history_msgs[-1]["content"] = acc
                yield ui_return(
                    gr.update(value="") if GRADIO_AVAILABLE else "",
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    acc,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
            return
        
        # ===== BƯỚC 3: XỬ LÝ FETCH / RAG_SEARCH / HYBRID =====
        docs: List[Dict[str, Any]] = []
        bm25_docs: List[Dict[str, Any]] = []
        emb_docs: List[Dict[str, Any]] = []
        
        if query_action == "fetch":
            # Fetch với filters
            app_log.info("🎯 FETCH mode", extra={"__kv__": {"filters": str(filters)}})
            docs = _fetch(filters, limit=int(k))
            
            # Fallback sang RAG nếu không tìm thấy
            if not docs:
                app_log.info("⚠️ Fetch trống → Fallback sang RAG")
                log_step("fetch_fallback")
                
                # Rewrite query trước khi RAG
                try:
                    search_query = rewrite_query(search_query)
                except Exception as e:
                    app_log.warning(f"Rewrite lỗi trong FETCH fallback: {e}, dùng original query")
                    search_query = message  # Fallback to original query
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bm25_docs, emb_docs, docs = loop.run_until_complete(
                        search_law(original_query=message, normalized_query=search_query, top_k=int(k))
                    )
                finally:
                    loop.close()
        
        elif query_action == "rag_search":
            # RAG search thuần
            app_log.info("🔍 RAG_SEARCH mode")
            
            # Rewrite query
            try:
                search_query = rewrite_query(search_query)
            except Exception as e:
                app_log.warning(f"Rewrite lỗi trong RAG_SEARCH: {e}, dùng original query")
                search_query = message  # Fallback to original query
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bm25_docs, emb_docs, docs = loop.run_until_complete(
                    search_law(original_query=message, normalized_query=search_query, top_k=int(k))
                )
            finally:
                loop.close()
        
        elif query_action == "hybrid":
            # Hybrid: Fetch + RAG
            app_log.info("⚡ HYBRID mode", extra={"__kv__": {"filters": str(filters)}})
            
            # Fetch trước
            fetch_docs = _fetch(filters, limit=int(k) // 2)
            
            # Rewrite query
            try:
                search_query = rewrite_query(search_query)
            except Exception as e:
                app_log.warning(f"Rewrite lỗi trong HYBRID: {e}, dùng original query")
                search_query = message  # Fallback to original query
            
            # RAG search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bm25_docs, emb_docs, rag_docs = loop.run_until_complete(
                    search_law(original_query=message, normalized_query=search_query, top_k=int(k) // 2)
                )
            finally:
                loop.close()
            
            # Merge docs (ưu tiên fetch)
            seen_ids = set()
            docs = []
            for d in fetch_docs:
                doc_id = f"{d.get('article_no')}_{d.get('clause_no')}_{d.get('point_letter')}"
                if doc_id not in seen_ids:
                    docs.append(d)
                    seen_ids.add(doc_id)
            
            for d in rag_docs:
                doc_id = f"{d.get('article_no')}_{d.get('clause_no')}_{d.get('point_letter')}"
                if doc_id not in seen_ids:
                    docs.append(d)
                    seen_ids.add(doc_id)
        
        # ===== BƯỚC 4: KIỂM TRA KẾT QUẢ =====
        if not docs:
            reply = "Chưa tìm thấy cơ sở pháp lý phù hợp. Bạn có thể bổ sung Điều/Khoản hoặc thêm bối cảnh."
            upd = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            yield ui_return(
                gr.update(value="") if GRADIO_AVAILABLE else "",
                upd,
                "(Chưa có dữ liệu)",
                "(Chưa có dữ liệu)",
                "(Chưa có dữ liệu)",
                reply,
                [],
                1,
                " Trang 0/0",
                upd,
            )
            return
        
        # ===== BƯỚC 5: SINH CÂU TRẢ LỜI =====
        bm25_markdown = docs_to_markdown(bm25_docs)
        emb_markdown = docs_to_markdown(emb_docs)
        cites_markdown, page_label = docs_page_markdown(docs, 1, int(cur_page_size))
        
        prompt = build_prompt(message, docs, history_msgs)
        log_step("llm_start", num_docs=len(docs), action=query_action)
        
        history_msgs = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        
        acc = ""
        yield ui_return(
            gr.update(value="") if GRADIO_AVAILABLE else "",
            history_msgs,
            bm25_markdown,
            emb_markdown,
            cites_markdown,
            acc,
            docs,
            1,
            page_label,
            history_msgs,
        )
        
        buffer = ""
        for chunk in stream_answer(prompt, temperature=float(temperature)):
            buffer += chunk
            if len(buffer) >= 50:
                acc += buffer
                history_msgs[-1]["content"] = acc
                yield ui_return(
                    gr.update(value="") if GRADIO_AVAILABLE else "",
                    history_msgs,
                    bm25_markdown,
                    emb_markdown,
                    cites_markdown,
                    acc,
                    docs,
                    1,
                    page_label,
                    history_msgs,
                )
                buffer = ""
        
        if buffer:
            acc += buffer
            history_msgs[-1]["content"] = acc
            yield ui_return(
                gr.update(value="") if GRADIO_AVAILABLE else "",
                history_msgs,
                bm25_markdown,
                emb_markdown,
                cites_markdown,
                acc,
                docs,
                1,
                page_label,
                history_msgs,
            )
        return
    
    except Exception as e:
        app_log.error(f"Lỗi xử lý: {e}")
        yield ui_return(
            gr.update(value="") if GRADIO_AVAILABLE else "",
            history_msgs,
            "(Lỗi hệ thống)",
            "(Lỗi hệ thống)",
            "(Lỗi hệ thống)",
            f"Lỗi: {e}",
            [],
            1,
            " Trang 0/0",
            history_msgs,
        )

def respond_wrapper(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
    # Note: threshold parameter kept for backward compatibility but not used
    for output in respond_generator(message, history_msgs, cur_page_size, k, temperature):
        yield output

# =========================
# BUILD UI
# =========================
def build_ui():
    if not GRADIO_AVAILABLE:
        raise RuntimeError("gradio không cài đặt")
    
    with gr.Blocks(title="⚖️ Trợ lý Luật Bất Động Sản", css=CSS) as demo:
        gr.Markdown("""
        ### ⚖️ Trợ lý Luật Bất Động Sản Việt Nam
        *Tra cứu văn bản • Giải thích dễ hiểu • Không thay thế luật sư*
        """)
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    value=[],
                    type="messages",
                    show_copy_button=True,
                    elem_id="chatbot",
                    autoscroll=True
                )
                with gr.Row():
                    ex1 = gr.Button("Chào bạn")
                    ex2 = gr.Button("Điều 10 Luật Đất đai quy định gì về quyền sử dụng đất")
                    ex3 = gr.Button("Quy trình chuyển nhượng đất là gì")
            
            with gr.Column(scale=5):
                gr.Markdown("**📜 Kết quả BM25**")
                bm25_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="bm25-box")
                
                gr.Markdown("**📜 Kết quả Embedding Search**")
                emb_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="emb-box")
                
                gr.Markdown("**Cơ sở pháp lý**")
                cites_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="cites-box")
                
                with gr.Row():
                    prev_page = gr.Button("⬅️")
                    next_page = gr.Button("➡️")
                
                with gr.Row():
                    page_info = gr.Markdown(" Trang 0/0")
                    page_size = gr.Slider(3, 20, value=5, step=1, label="Mỗi trang")
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Nhập câu hỏi về Luật Bất động sản...",
                scale=5,
                autofocus=False
            )
            send = gr.Button("Gửi", variant="primary", scale=1)
            clear = gr.Button("Làm mới", scale=1)
        
        def _fill(text):
            return text
        
        ex1.click(lambda: _fill("Chào bạn"), outputs=msg)
        ex2.click(lambda: _fill("Điều 10 Luật Đất đai quy định gì về quyền sử dụng đất"), outputs=msg)
        ex3.click(lambda: _fill("Quy trình chuyển nhượng đất là gì"), outputs=msg)
        
        state_history = gr.State([])
        state_last_answer = gr.State("")
        state_docs = gr.State([])
        state_page = gr.State(1)
        
        outputs = [
            msg, chatbot, bm25_md, emb_md, cites_md,
            state_last_answer, state_docs, state_page, page_info, state_history
        ]
        
        send.click(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
        msg.submit(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
        
        def on_like(data: gr.LikeData):
            msg_like = data.value or {}
            role = msg_like.get("role", "assistant")
            text = msg_like.get("content", "")
            app_log.info(
                "Phản hồi người dùng",
                extra={"__kv__": {"thich": data.liked, "vai_tro": role, "do_dai": len(text or "")}}
            )
            return None
        
        chatbot.like(on_like)
        
        def render_cites_for_page(docs, page, cur_page_size):
            md, label = docs_page_markdown(docs or [], int(page), int(cur_page_size))
            return gr.update(value=md), int(page), label
        
        def go_prev(docs, page, cur_page_size):
            if not docs:
                return render_cites_for_page([], 1, cur_page_size)
            new_page = max(1, int(page) - 1)
            return render_cites_for_page(docs, new_page, cur_page_size)
        
        def go_next(docs, page, cur_page_size):
            if not docs:
                return render_cites_for_page([], 1, cur_page_size)
            _, total, total_pages, _ = paginate_docs(docs, 1, int(cur_page_size))
            new_page = min(total_pages if total_pages > 0 else 1, int(page) + 1)
            return render_cites_for_page(docs, new_page, cur_page_size)
        
        def on_change_page_size(docs, cur_page_size):
            return render_cites_for_page(docs, 1, cur_page_size)
        
        prev_page.click(
            go_prev,
            inputs=[state_docs, state_page, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False
        )
        next_page.click(
            go_next,
            inputs=[state_docs, state_page, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False
        )
        page_size.release(
            on_change_page_size,
            inputs=[state_docs, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False
        )
        
        gr.Markdown(
            f"<sub>© {datetime.now().year} — Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.</sub>"
        )
    
    return demo

# =========================
# RUN
# =========================
if __name__ == "__main__":
    RUN_UI = os.getenv("RUN_UI", "1").strip().lower() not in {"0", "false", "no"}
    
    if RUN_UI and GRADIO_AVAILABLE:
        demo = build_ui()
        demo.queue()
        demo.launch(show_error=True, share=False)
    else:
        print("Chạy ở chế độ CLI (không chạy Gradio UI). Gõ câu hỏi để thử (Ctrl+C để thoát).")
        history = []
        try:
            while True:
                q = input("Bạn: ").strip()
                if not q:
                    continue
                for out in respond_generator(q, history, cur_page_size=5):
                    msg_val, chat_val, bm25_val, emb_val, cites_val, last_answer_val, docs_val, page_val, page_label_val, history_msgs = out
                    if last_answer_val:
                        print("\n--- Trợ lý đang trả lời ---")
                        print(last_answer_val)
        except KeyboardInterrupt:
            print("\nThoát.")