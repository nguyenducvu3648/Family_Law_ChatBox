import asyncio
import os
import re
import json
import time
import logging
from datetime import datetime
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential

# ================== CẤU HÌNH MÔI TRƯỜNG ==================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash") # Đã đổi sang 1.5 flash
INTENT_DEBUG = os.getenv("INTENT_DEBUG", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CASUAL_MAX_WORDS = int(os.getenv("CASUAL_MAX_WORDS", "0").strip() or 0)  # 0 = không giới hạn
INTENT_RAW_PREVIEW_LIMIT = int(os.getenv("INTENT_RAW_PREVIEW_LIMIT", "240").strip() or 240)
INTENT_FALLBACK_CASUAL = os.getenv(
"INTENT_FALLBACK_CASUAL",
"Chào bạn, mình có thể hỗ trợ câu hỏi về Luật Hôn nhân & Gia đình. Bạn muốn hỏi nội dung gì?",
).strip()

if not (QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong tệp .env")

# ================== THIẾT LẬP LOGGING ==================

# SỬA LỖI: Thêm định nghĩa KVFormatter
class KVFormatter(logging.Formatter):
    """Formatter để thêm các cặp key=value từ record.extra"""
    def format(self, record):
        # Format a base message
        msg = super().format(record)
        # Check if there's a __kv__ extra
        if hasattr(record, "__kv__") and record.__kv__:
            kv_pairs = " ".join(f"{k}={v}" for k, v in record.__kv__.items())
            # Thêm vào cuối message, phân tách bằng |
            msg = f"{msg} | {kv_pairs}"
        return msg

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
handlers = [
    logging.StreamHandler(),
    logging.FileHandler("botchat_honnhan.log", encoding="utf-8"),
]
for h in handlers:
    h.setFormatter(KVFormatter(LOG_FORMAT)) # Sử dụng KVFormatter

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
for h in handlers:
    root_logger.addHandler(h)

app_log = logging.getLogger("app")
metrics_logger = logging.getLogger("metrics")
if not metrics_logger.handlers:
    fh = logging.FileHandler("metrics.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(fh)
    metrics_logger.setLevel(logging.INFO)

def log_step(event: str, **kv):
    kvpairs = ",".join([f"{k}={v}" for k, v in kv.items()])
    metrics_logger.info(f"ts={int(time.time())},evt={event},{kvpairs}")

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
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            app_log.info(
                f"Thời gian thực thi {func.__name__}",
                extra={"__kv__": {"thoi_gian": f"{elapsed:.4f} giây"}},
            )
            
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# ================== KHỞI TẠO ==================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

genai.configure(api_key=GEMINI_API_KEY)
INTENT_SYSTEM_PROMPT = dedent("""
Bạn là trợ lý về Luật Hôn nhân & Gia đình Việt Nam.
Trả về **JSON thuần** (không markdown, không lời dẫn).

Schema một trong các dạng:
1) {"intent":"casual","answer":"..."}
2) {"intent":"legal_answer","normalized_query":"...","original_query":"..."}
3) {"intent":"law_search","filters":{"article_no":int?,"clause_no":int?,"point_letter":str?,"chapter_number":int?}}

Quy tắc xác định intent:
- Hỏi về điều/khoản/chương/mục cụ thể → law_search.
- Hỏi xã giao/chào hỏi → casual.
- Nhắc số điều/khoản nhưng hỏi tình huống thực tế, áp dụng, thủ tục → legal_answer.
- Luôn dựa vào **mục đích câu hỏi**, không chỉ dựa vào số điều/khoản.

Nếu intent = casual thì bắt buộc có answer (tiếng Việt, lịch sự).
""")
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
    system_instruction=INTENT_SYSTEM_PROMPT,
)
answer_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID)

def _safe_truncate(text: str, limit: int = 800) -> str:
    return text if text and len(text) <= limit else (text[:limit] + "…(cắt)") if text else ""

LEGAL_HINTS = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|hôn nhân|ly hôn|ly thân|nuôi con|tài sản|"
    r"quan hệ vợ chồng|kết hôn|hủy kết hôn|chung sống như vợ chồng|cấp dưỡng|giám hộ)\b"
)

def looks_like_legal(query: str) -> bool:
    return bool(LEGAL_HINTS.search(query or ""))

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

# @log_time
def encode_query(text: str):
    key = f"{EMBEDDING_MODEL}|query|{text}"
    v = embed_cache.get(key)
    if v is not None:
        return v
    vec = embedder.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()
    embed_cache.set(key, vec)
    return vec

# ================== PHÂN TÍCH Ý ĐỊNH (INTENT) ==================
# @log_time
def _intent_via_gemini(query: str) -> Dict[str, Any]:
    try:
        cfg = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=192,
            response_mime_type="application/json",
        )
        resp = gemini_model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [f"Câu hỏi: {query}\nHãy trả JSON thuần phù hợp schema đã nêu."],
                }
            ],
            generation_config=cfg,
        )

        candidates = getattr(resp, "candidates", None) or []
        first_cand = candidates[0] if candidates else None
        finish_reason = getattr(first_cand, "finish_reason", None)
        safety = []
        try:
            if first_cand and getattr(first_cand, "safety_ratings", None):
                for s in first_cand.safety_ratings:
                    cat = getattr(s, "category", "")
                    prob = getattr(s, "probability", "")
                    safety.append(f"{cat}:{prob}")
        except Exception:
            pass

        raw = getattr(resp, "text", "") or ""
        app_log.info(
            "Kết quả phân tích ý định",
            extra={
                "__kv__": {
                    "do_dai": len(raw),
                    "xem_truoc": _safe_truncate(raw, INTENT_RAW_PREVIEW_LIMIT),
                    "so_ung_vien": len(candidates),
                    "ly_do_ket_thuc": finish_reason,
                    "bao_mat": ";".join(safety[:6]),
                }
            },
        )

        if finish_reason == 2 or not raw:
            log_step("intent_block", ly_do=str(finish_reason))
            app_log.warning(
                "Phân tích ý định bị chặn",
                extra={"__kv__": {"ly_do_ket_thuc": finish_reason, "do_dai_raw": len(raw)}}
            )
            return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            app_log.warning("Kết quả phân tích không phải dict")
            return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}
        out: Dict[str, Any] = {}
        for k in ("intent", "answer", "normalized_query", "filters", "original_query"):
            if k in data and data[k] not in (None, ""):
                out[k] = data[k]
        app_log.info(
            "Ý định đã phân tích",
            extra={
                "__kv__": {
                    "loai_y_dinh": out.get("intent", ""),
                    "co_tra_loi": int("answer" in out and bool(out.get("answer"))),
                    "co_bo_loc": int("filters" in out and bool(out.get("filters"))),
                    "do_dai_tra_loi": len(out.get("answer", "") or ""),
                    "ly_do_ket_thuc": finish_reason,
                }
            },
        )
        return out
    except Exception as e:
        app_log.warning("Lỗi phân tích ý định", extra={"__kv__": {"loi": str(e)}})
        return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

# @log_time
def analyze_intent(query: str) -> Dict[str, Any]:
    data = _intent_via_gemini(query)
    intent = data.get("intent")
    answer = data.get("answer", "")
    normalized_query = data.get("normalized_query", "") or query
    original_query = data.get("original_query", "")
    filters = data.get("filters", {}) or {}

    if intent not in {"casual", "legal_answer", "law_search"}:
        if re.search(r"(?i)\bđiều\s*\d+", query) or re.search(r"(?i)\bkhoản\s*\d+", query):
            intent = "law_search"
        elif looks_like_legal(query):
            intent = "legal_answer"
        else:
            intent = "casual"
        log_step("intent_fallback", do_dai_query=len(query))

    log_step("intent", loai=intent, co_legal=str(looks_like_legal(query)))
    app_log.info("Quyết định ý định", extra={"__kv__": {"loai_y_dinh": intent}})
    return {
        "intent": intent,
        "answer": answer,
        "normalized_query": normalized_query,
        "original_query": original_query,
        "filters": filters,
    }

# ================== TÌM KIẾM VECTOR ==================
# @log_time
def _build_filter(query_text: str) -> Optional[Filter]:
    conds: List[FieldCondition] = []
    m = re.search(r"(?i)\bđiều\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="article_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bkhoản\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="clause_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bđiểm\s*([a-z])\b", query_text)
    if m:
        conds.append(FieldCondition(key="point_letter", match=MatchValue(value=m.group(1).lower())))
    m = re.search(r"(?i)\bchương\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="chapter_number", match=MatchValue(value=int(m.group(1)))))
    return Filter(must=conds) if conds else None

@log_time
async def search_law(query: str, top_k: int = 15, score_threshold: float = 0.42):
    """Thực hiện vector search (đã loại bỏ BM25 và Reranker)"""
    t0 = time.perf_counter()
    app_log.info(
        "Bắt đầu tìm kiếm vector",
        extra={"__kv__": {"cau_hoi": _safe_truncate(query, 80), "top_k": top_k, "nguong_diem": score_threshold}},
    )

    cache_key = f"search_vector_only|{COLLECTION_NAME}|{top_k}|{score_threshold}|{query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("Tìm kiếm vector từ bộ nhớ cache")
        return cached

    try:
        # 1. Xây dựng bộ lọc (nếu có)
        flt = _build_filter(query)
        
        # 2. Mã hóa câu hỏi
        t_embed0 = time.perf_counter()
        vec = encode_query(query)
        t_embed = time.perf_counter() - t_embed0

        # 3. Truy vấn Qdrant
        t_q0 = time.perf_counter()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            with_payload=True,
            limit=int(top_k),         # Chỉ lấy top_k
            query_filter=flt,         # Áp dụng bộ lọc
            score_threshold=float(score_threshold) # Áp dụng ngưỡng điểm
        )
        t_qdrant = time.perf_counter() - t_q0
        
        # 4. Định dạng kết quả
        selected_docs = []
        for r in results.points:
            p = r.payload or {}
            selected_docs.append({
                "source_id": p.get("source_id", ""), # SỬA LỖI: Thêm source_id
                "citation": p.get("exact_citation", ""),
                "chapter_number": p.get("chapter_number", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": float(r.score or 0.0), # Giữ 'score' để markdown dùng
            })

        search_cache.set(cache_key, selected_docs)
        
        sk_top1 = selected_docs[0]['score'] if selected_docs else 0.0
        
        log_step(
            "tim_kiem_vector",
            k_yeu_cau=top_k,
            k_tra_ve=len(selected_docs),
            diem_top1=f"{sk_top1:.4f}",
            t_nhung=f"{t_embed:.4f}",
            t_qdrant=f"{t_qdrant:.4f}",
            t_tong=f"{time.perf_counter()-t0:.4f}",
        )
        app_log.info(
            "Tìm kiếm vector hoàn tất",
            extra={"__kv__": {"so_luong": len(selected_docs), "diem_top1": f"{sk_top1:.4f}"}})
        
        # Trả về danh sách tài liệu duy nhất
        return selected_docs
        
    except Exception as e:
        app_log.error("Lỗi tìm kiếm vector", extra={"__kv__": {"loi": str(e)}})
        log_step("tim_kiem_loi", thong_bao=str(e))
        raise

# ================== CÔNG CỤ RENDER ==================
def law_line(d: Dict[str, Any]) -> Tuple[str, str, str]:
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
    return cited, chapter, title

def docs_to_markdown(docs: List[Dict[str, Any]]):
    if not docs:
        return "❌ Không tìm thấy điều luật nào."
    lines = []
    for i, d in enumerate(docs, 1):
        cited, chapter, title = law_line(d)
        content = (d.get("content") or "").strip()
        score = round(d.get("score", 0.0), 4)
        sid = d.get("source_id", "")
        lines.append(
            f"**{i}. [{sid}] {cited}{chapter}{title}** \n" +
            f"{content}  \n" +
            f"<sub>Độ liên quan: {score}</sub>\n"
        )
    return "\n".join(lines)

def paginate_docs(docs, page: int, page_size: int):
    total = len(docs)
    if total == 0:
        return [], 0, 0, 0
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    start = (page - 1) * page_size
    end = start + page_size
    sliced = docs[start:end]
    total_pages = (total + page_size - 1) // page_size
    return sliced, total, total_pages, start


def docs_page_markdown(docs, page: int, page_size: int):
    sliced, total, total_pages, start = paginate_docs(docs, page, page_size)
    if total == 0:
        return "(Chưa có dữ liệu)", " Trang 0/0"
    body = docs_to_markdown(sliced)
    page_label = f" Trang {page}/{total_pages} — hiển thị {start+1}–{min(start+len(sliced), total)} / {total}"
    return f"**{page_label}**\n\n{body}", page_label

# ================== XÂY DỰNG PROMPT ==================
# @log_time
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

    docs_sorted = sorted(
        docs,
        key=lambda d: (
            int(d.get("article_no") or 9999),
            int(d.get("clause_no") or 9999),
            str(d.get("point_letter") or ""),
        ),
    )

    context_lines = []
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

    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."

    prompt = dedent(f"""
    Bạn là luật sư tư vấn Luật Hôn nhân & Gia đình, chỉ dùng trích đoạn trong danh sách sau.
    Quy tắc:
    - Câu hỏi Đúng/Sai → trả lời **Kết luận: Đúng/Sai** + lý do.
    - Câu hỏi thường → trả lời **1–3 câu**, bám sát câu hỏi.
    - **Trích dẫn nguyên văn** điều luật liên quan (Điểm–Khoản–Điều + nội dung), theo thứ tự.
    - Nếu thiếu căn cứ → trả lời: **Không đủ căn cứ.**
    - Câu hỏi ngoài luật → trả lời lịch sự, ngắn gọn, không viện dẫn luật.
    ĐỊNH DẠNG TRẢ LỜI:
    - Trích dẫn: <liệt kê toàn bộ Điểm–Khoản–Điều + nội dung>
    - Giải thích: <1–3 câu, áp dụng tình huống>
    - Kết luận: <kết luận ngắn gọn dựa vào câu hỏi và giải thích>

    Câu hỏi hiện tại:
    \"\"\"{query}\"\"\"{history_block}

    Danh sách điều luật (top_k):
    {context}
    """).strip()

    return prompt

# ================== XỬ LÝ TRẢ LỜI LLM ==================
# SỬA LỖI: Kích hoạt lại @retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _gemini_stream_async(prompt, temperature: float):
    """SỬA LỖI: Chuyển sang phiên bản async"""
    cfg = genai.types.GenerationConfig(temperature=float(temperature))
    # SỬA LỖI: Sử dụng generate_content_async
    return await answer_model.generate_content_async(prompt, generation_config=cfg, stream=True)

@log_time
async def stream_answer(prompt, temperature=0.2):
    """SỬA LỖI: Chuyển sang async def để dùng 'async for'"""
    t0 = time.perf_counter()
    t_first0 = time.perf_counter()
    first_token_emitted = False
    try:
        # SỬA LỖI: Xóa cuộc gọi trùng lặp
        resp = await _gemini_stream_async(prompt, temperature)
        
        # SỬA LỖI: Dùng 'async for'
        async for ch in resp:
            if getattr(ch, "text", None):
                if not first_token_emitted:
                    log_step("llm_first_token", thoi_gian_truoc=f"{time.perf_counter()-t_first0:.4f}")
                    first_token_emitted = True
                yield ch.text
    except Exception as e:
        app_log.error("Lỗi gọi mô hình LLM", extra={"__kv__": {"loi": str(e)}})
        yield f"\n\nLỗi gọi mô hình: {e}"
    finally:
        log_step("llm_tong", thoi_gian=f"{time.perf_counter()-t0:.4f}")

# ================== LẤY DỮ LIỆU QDRANT ==================
# @log_time
def _fetch(filters: Dict[str, Any], limit: int = 10):
    must = []
    mapping = {"article_no": int, "clause_no": int, "point_letter": str, "chapter_number": int}
    for k, caster in mapping.items():
        if k in filters and filters[k] not in (None, ""):
            try:
                val = caster(filters[k])
                must.append(FieldCondition(key=k, match=MatchValue(value=val)))
            except Exception as e:
                app_log.warning(
                    "Lỗi ép kiểu dữ liệu bộ lọc",
                    extra={"__kv__": {"truong": k, "gia_tri": filters[k], "loi": str(e)}},
                )
                try:
                    val = str(filters[k])
                    must.append(FieldCondition(key=k, match=MatchValue(value=val)))
                except:
                    pass
    app_log.info(
        "Bộ lọc tìm kiếm",
        extra={"__kv__": {"bo_loc_goc": str(filters), "dieu_kien_must": str(must)}},
    )
    if not must:
        app_log.warning("Không có điều kiện must")
        return []
    flt = Filter(must=must)
    out = []
    try:
        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=min(64, max(5, limit)),
            with_payload=True,
        )
        app_log.info(
            "Kết quả tìm kiếm Qdrant",
            extra={"__kv__": {"so_luong": len(scroll_res), "collection": COLLECTION_NAME}},
        )
        for r in scroll_res:
            p = r.payload or {}
            out.append({
                "source_id": p.get("source_id", ""), # SỬA LỖI: Thêm source_id
                "chapter": p.get("chapter", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": 1.0,
            })
            if len(out) >= limit:
                break
    except Exception as e:
        app_log.error(
            "Lỗi khi tìm kiếm Qdrant",
            extra={"__kv__": {"loi": str(e), "collection": COLLECTION_NAME}},
        )
        return []
    if not out:
        app_log.warning("Không tìm thấy kết quả", extra={"__kv__": {"bo_loc": str(filters)}})
    return out

# ================== HÀM HỖ TRỢ GIAO DIỆN ==================
def ui_return(msg_val, chatbot_val, emb_val, cites_val, last_answer_val, docs_val, page_val, page_label_val, history_msgs):
    """Helper function đảm bảo luôn trả về đúng 9 giá trị (đã xóa bm25)"""
    print("DEBUG: Gọi hàm ui_return, trả về 9 giá trị")
    return (
        msg_val,
        chatbot_val,
        gr.update(value=emb_val),
        gr.update(value=cites_val),
        last_answer_val,
        docs_val,
        page_val,
        page_label_val,
        history_msgs,
    )

# ================== GIAO DIỆN NGƯỜI DÙNG ==================
CSS = """
#chatbot { height: 540px !important; }
label { font-size:12px !important; opacity:.9 }
#cites-box {
    max-height: 480px; /* Tăng chiều cao */
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
#emb-box {
    max-height: 280px; /* Tăng chiều cao */
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
"""

with gr.Blocks(
    title="⚖️ Trợ lý Luật Hôn Nhân & Gia Đình 2014",
    css=CSS,
) as demo:
    gr.Markdown("""
    ### ⚖️ Trợ lý Luật Hôn Nhân & gia đình 2014
    *Tham chiếu chính xác • Hạn chế suy diễn • Không thay thế tư vấn pháp lý*
    """)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                show_copy_button=True,
                elem_id="chatbot",
                autoscroll=True,
            )
            with gr.Row():
                ex1 = gr.Button("Chào bạn")
                ex2 = gr.Button("Điều 81 quy định gì về việc nuôi con sau ly hôn")
                ex3 = gr.Button("Khoản 2 Điều 56 nói gì")
        with gr.Column(scale=5):
            # ĐÃ XÓA PANEL BM25
            gr.Markdown("**📜 Kết quả tìm kiếm (Vector Search)**")
            emb_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="emb-box")
            
            gr.Markdown("**Cơ sở pháp lý (Chi tiết)**")
            cites_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="cites-box")
            with gr.Row():
                prev_page = gr.Button("⬅️")
                next_page = gr.Button("➡️")
            with gr.Row():
                page_info = gr.Markdown(" Trang 0/0")
                page_size = gr.Slider(3, 20, value=5, step=1, label="Mỗi trang")

    with gr.Row():
        msg = gr.Textbox(placeholder="Nhập câu hỏi...", scale=5, autofocus=False)
        send = gr.Button("Gửi", variant="primary", scale=1)
        clear = gr.Button("Làm mới", scale=1)

    # Điền sẵn ví dụ
    def _fill(text):
        return text

    ex1.click(lambda: _fill("Chào bạn"), outputs=msg)
    ex2.click(
        lambda: _fill("Điều 81 quy định gì về việc nuôi con sau ly hôn"),
        outputs=msg,
    )
    ex3.click(lambda: _fill("Khoản 2 Điều 56 nói gì"), outputs=msg)

    # Trạng thái
    state_history = gr.State([])
    state_last_answer = gr.State("")
    state_docs = gr.State([])
    state_page = gr.State(1)

    # -------- GENERATOR CHÍNH - SỬA LỖI: Chuyển sang async def --------
    @log_time
    async def respond_generator(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
        """Generator function - yield từng update dần dần (ĐÃ CHUYỂN SANG ASYNC)"""
        print(f"DEBUG: Bắt đầu xử lý câu hỏi: {message}")
        if not (message and message.strip()):
            print("DEBUG: Câu hỏi rỗng, trả về mặc định")
            gr.Info("Vui lòng nhập câu hỏi.")
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                # "", # Bỏ bm25_val
                "",
                "",
                "",
                [],
                1,
                " Trang 0/0",
                history_msgs,
            )
            return

        t_overall0 = time.perf_counter()
        try:
            # Phân tích ý định
            print("DEBUG: Gọi hàm phân tích ý định")
            intent_info = analyze_intent(message)
            print(f"DEBUG: Kết quả ý định: {intent_info}")
            intent = intent_info["intent"]
            intent_answer = intent_info.get("answer", "")
            normalized_query = intent_info.get("normalized_query", message)
            original_query = intent_info.get("original_query", message)
            intent_filters = intent_info.get("filters", {})

            # ========== XỬ LÝ CÂU HỎI XÃ GIAO ==========
            if intent == "casual":
                final_answer = (intent_answer or "").replace("\u200b", "").strip()
                app_log.info(
                    "Xử lý câu hỏi xã giao",
                    extra={"__kv__": {"do_dai_tra_loi": len(final_answer)}},
                )

                if final_answer and CASUAL_MAX_WORDS > 0:
                    words = final_answer.split()
                    if len(words) > CASUAL_MAX_WORDS:
                        truncated = " ".join(words[:CASUAL_MAX_WORDS])
                        app_log.info(
                            "Cắt ngắn câu trả lời xã giao",
                            extra={
                                "__kv__": {
                                    "so_tu_goc": len(words),
                                    "so_tu_giu": CASUAL_MAX_WORDS,
                                    "do_dai_goc": len(final_answer),
                                }
                            },
                        )
                        final_answer = truncated

                # Nếu có câu trả lời trực tiếp từ intent
                if len(final_answer) >= 1:
                    history_msgs = history_msgs + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final_answer},
                    ]
                    print("DEBUG: Trả về câu trả lời xã giao trực tiếp")
                    yield ui_return(
                        gr.update(value=""),
                        history_msgs,
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        final_answer,
                        [],
                        1,
                        " Trang 0/0",
                        history_msgs,
                    )
                    return

                # Stream câu trả lời xã giao
                simple_prompt = "Trả lời thân thiện ngắn gọn (<=2 câu) tiếng Việt cho câu: " + message
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": ""},
                ]
                acc = ""
                print("DEBUG: Bắt đầu stream câu trả lời xã giao")
                
                # Yield initial state
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    acc,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
                
                # Stream từng chunk - SỬA LỖI: dùng async for và bỏ buffer
                async for chunk in stream_answer(simple_prompt, temperature=float(temperature)):
                    acc += chunk
                    history_msgs[-1]["content"] = acc
                    yield ui_return(
                        gr.update(value=""),
                        history_msgs,
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        acc,
                        [],
                        1,
                        " Trang 0/0",
                        history_msgs,
                    )

                print("DEBUG: Hoàn thành stream câu trả lời xã giao")
                return

            # ========== XỬ LÝ CÂU HỎI PHÁP LÝ ==========
            docs: List[Dict[str, Any]] = []
            source = None

            if intent == "law_search":
                print("DEBUG: Tìm kiếm điều luật")
                docs = _fetch(intent_filters, limit=int(k)) if intent_filters else []
                source = "law_search_fetch"
                if not docs:
                    app_log.info(
                        "Rơi vào tìm kiếm vector (fallback)",
                        extra={"__kv__": {"cau_hoi": message}},
                    )
                    # SỬA LỖI: Dùng await trực tiếp
                    docs = await search_law(
                        message, top_k=int(k), score_threshold=float(threshold)
                    )
                    source = "law_search_vector_fallback"

            elif intent == "legal_answer":
                print("DEBUG: Tìm kiếm câu trả lời pháp lý (vector search)")
                # SỬA LỖI: Dùng await trực tiếp
                docs = await search_law(
                    normalized_query, top_k=int(k), score_threshold=float(threshold)
                )
                source = "legal_answer_vector"

            else:
                reply = INTENT_FALLBACK_CASUAL
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
                print("DEBUG: Trả về ý định mặc định")
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    reply,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
                return

            if not docs:
                reply = (
                    "Chưa tìm thấy cơ sở pháp lý phù hợp. "
                    "Bạn có thể bổ sung Điều/Khoản hoặc thêm bối cảnh."
                )
                upd = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
                print("DEBUG: Không tìm thấy tài liệu")
                yield ui_return(
                    gr.update(value=""),
                    upd,
                    "(Chưa có dữ liệu)",
                    "(Chưa có dữ liệu)",
                    reply,
                    [],
                    1,
                    " Trang 0/0",
                    upd,
                )
                return

            # Chuẩn bị prompt và markdown
            if intent == "legal_answer":
                user_query = original_query or message
            elif intent == "law_search":
                user_query = message
            else:
                user_query = message
            
            # ĐÃ XÓA bm25_markdown
            emb_markdown = docs_to_markdown(docs) # emb_md giờ là docs chính
            cites_markdown, page_label = docs_page_markdown(docs, 1, int(cur_page_size))
            prompt = build_prompt(user_query, docs, history_msgs)

            log_step("llm_chuanbi", so_tai_lieu=len(docs), nguon=source)
            print(f"DEBUG: Đã chuẩn bị prompt, số tài liệu: {len(docs)}")

            # Chuẩn bị history
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            acc = ""
            
            print("DEBUG: Bắt đầu stream câu trả lời pháp lý")
            
            # Yield initial state với các markdown
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                emb_markdown, # Hiển thị docs lên panel emb
                cites_markdown,
                acc,
                docs,
                1,
                page_label,
                history_msgs,
            )
            
            # Stream từng chunk - SỬA LỖI: dùng async for và bỏ buffer
            async for chunk in stream_answer(prompt, temperature=float(temperature)):
                acc += chunk
                history_msgs[-1]["content"] = acc
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    emb_markdown,
                    cites_markdown,
                    acc,
                    docs,
                    1,
                    page_label,
                    history_msgs,
                )

            print("DEBUG: Hoàn thành stream câu trả lời pháp lý")
            return

        except Exception as e:
            app_log.error("Lỗi xử lý câu hỏi", extra={"__kv__": {"loi": str(e)}})
            print(f"DEBUG: Lỗi trong xử lý: {e}")
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                "(Lỗi hệ thống)",
                "(Lỗi hệ thống)",
                f"Lỗi: {e}",
                [],
                1,
                " Trang 0/0",
                history_msgs,
            )
            return

    # -------- WRAPPER BẤT ĐỒNG BỘ - Gradio gọi hàm này --------
    async def respond_wrapper(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
        """SỬA LỖI: Wrapper async để Gradio gọi"""
        async for output in respond_generator(message, history_msgs, cur_page_size, k, temperature, threshold):
            yield output

    # Kết nối outputs (9 giá trị, đã xóa bm25_md)
    outputs = [
        msg,                  # 1
        chatbot,              # 2
        emb_md,               # 3
        cites_md,             # 4
        state_last_answer,    # 5
        state_docs,           # 6
        state_page,           # 7
        page_info,            # 8
        state_history,        # 9
    ]
    
    # Kết nối với wrapper (BẬT queue=True để hỗ trợ streaming)
    send.click(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
    msg.submit(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)

    # Nút Làm mới
    def clear_all():
        return (
            "",                     # msg
            [],                     # chatbot
            "(Chưa có dữ liệu)",    # emb_md
            "(Chưa có dữ liệu)",    # cites_md
            "",                     # state_last_answer
            [],                     # state_docs
            1,                      # state_page
            " Trang 0/0",           # page_info
            []                      # state_history
        )
    clear.click(clear_all, outputs=outputs, queue=False)

    # Like/Dislike
    def on_like(data: gr.LikeData):
        msg_like = data.value or {}
        role = msg_like.get("role", "assistant")
        text = msg_like.get("content", "")
        app_log.info(
            "Phản hồi người dùng",
            extra={"__kv__": {"thich": data.liked, "vai_tro": role, "do_dai": len(text or "")}},
        )
        return None
    chatbot.like(on_like)

    # Phân trang
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

    prev_page.click(go_prev, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
    next_page.click(go_next, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
    page_size.release(on_change_page_size, inputs=[state_docs, page_size], outputs=[cites_md, state_page, page_info], queue=False)

    gr.Markdown(f"""
    <sub>© {datetime.now().year} — Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.</sub>
    """)

if __name__ == "__main__":
    demo.queue()
    # Bạn có thể chạy demo.launch() với share=True nếu cần
    demo.launch(show_error=True)