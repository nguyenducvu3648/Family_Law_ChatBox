import os
import re
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# CONFIG (tương đương core.config)
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "hybrid-BDS"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

INTENT_DEBUG = os.getenv("INTENT_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CASUAL_MAX_WORDS = int(os.getenv("CASUAL_MAX_WORDS", "0").strip() or 0)
INTENT_RAW_PREVIEW_LIMIT = int(os.getenv("INTENT_RAW_PREVIEW_LIMIT", "240").strip() or 240)
INTENT_FALLBACK_CASUAL = os.getenv(
    "INTENT_FALLBACK_CASUAL",
    "Chào bạn, mình có thể hỗ trợ câu hỏi về Luật Bất động sản. Bạn muốn hỏi nội dung gì?",
).strip()

# =========================
# LOGGING SETUP (tương đương logging_setup.py)
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
# UTILS (tương đương utils.py)
# =========================
LEGAL_HINTS = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|đất|nhà ở|đầu tư|xây dựng|kinh doanh bất động sản|thuế|sổ đỏ|sổ hồng|quy hoạch|cho thuê|mua bán|chuyển nhượng)\b"
)

def looks_like_legal(query: str) -> bool:
    return bool(LEGAL_HINTS.search(query or ""))

def _safe_truncate(text: str, limit: int = 800) -> str:
    return text if text and len(text) <= limit else (text[:limit] + "…(cắt)") if text else ""

def normalize_legal_query(query: str) -> dict:
    """
    Normalize query to be more suitable for intent detection and retrieval.
    Kept from original but adapted generically.
    """
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

    intent = "general_legal"
    if re.search(r"\b(phân biệt|so sánh|khác nhau|giữa)\b", text, flags=re.IGNORECASE):
        intent = "compare"
    elif re.search(r"\b(phải|có|được|bị|nên)\b", text, flags=re.IGNORECASE):
        intent = "true_false"
    elif re.search(
        r"\b(nếu|trường hợp|giả sử|muốn hỏi|muốn biết|nên|có nên|làm sao|cách nào|xử lý ra sao|khởi kiện|hòa giải)\b",
        text,
        flags=re.IGNORECASE,
    ):
        intent = "advice"
    elif re.search(r"\b(là gì|định nghĩa|được hiểu như thế nào)\b", text, flags=re.IGNORECASE):
        intent = "definition"
    elif re.search(r"\b(mức phạt|xử phạt|phạt tiền|chế tài)\b", text, flags=re.IGNORECASE):
        intent = "punishment"
    elif re.search(r"\b(theo luật|theo quy định|căn cứ)\b", text, flags=re.IGNORECASE):
        intent = "law_reference"

    if intent == "general_legal":
        if re.search(r"\b(là|thuộc|bao gồm|gồm|được coi là|có nghĩa là|được xác định là)\b", text, flags=re.IGNORECASE):
            intent = "true_false"
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"

    if not text.endswith("?"):
        if intent == "true_false":
            if not re.search(r"\b(phải không|có đúng không|đúng không|được không)\b", text, flags=re.IGNORECASE):
                text += " phải không?"
        elif intent == "advice":
            text += "?"
        elif intent == "law_reference":
            if not re.search(r"\btheo quy định\b", text, flags=re.IGNORECASE):
                text += " theo quy định pháp luật?"
            else:
                text += "?"
        elif intent == "punishment":
            text += " bị xử lý thế nào?"
        else:
            text += "?"

    text = re.sub(r"[!?]{2,}", "?", text)
    text = text.replace(",,", ",").replace("..", ".")
    text = text.strip()

    return {
        "normalized_query": text,
        "intent_hint": intent,
        "original_query": original
    }

# =========================
# CACHE (tương đương cache.py)
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
# PROMPTS (system, intent, answer) - chuyển sang BĐS
# =========================
INTENT_TXT = r'''
Bạn là trợ lý về LUẬT BẤT ĐỘNG SẢN Việt Nam (bao gồm Luật Đất đai, Luật Nhà ở, Luật Xây dựng, Luật Đầu tư, Luật Kinh doanh BĐS, và các quy định về thuế sử dụng đất).
Trả về JSON thuần (không markdown, không lời dẫn).

Schema một trong các dạng:
1) {"intent":"casual","answer":"..."}
2) {"intent":"legal_answer","normalized_query":"...","original_query":"..."}
3) {"intent":"law_search","query_type":"fetch|compare|definition","filters":{"article_no":int?,"clause_no":int?,"point_letter":str?,"chapter_number":int?},"normalized_query":"...","original_query":"..."}

Quy tắc xác định intent:
- Chỉ hỏi về điều/khoản/chương/điểm cụ thể → law_search với query_type="fetch" và điền filters.
- So sánh/phân biệt các điều khoản hoặc khái niệm (ví dụ: "phân biệt quyền sử dụng đất và quyền sở hữu nhà", "so sánh bán nhà trả góp và hợp đồng mua bán") → law_search với query_type="compare".
- Giải thích khái niệm pháp lý (ví dụ: "quyền sử dụng đất là gì?", "sổ đỏ khác sổ hồng như thế nào?") → law_search với query_type="definition".
- Giao tiếp thông thường, chào hỏi → casual.
- Các câu hỏi tình huống, áp dụng luật, thủ tục, xin tư vấn → legal_answer.
- Luôn ưu tiên mục đích thực sự của câu hỏi, không chỉ dựa vào từ khóa.

Với law_search:
- Nếu hỏi điều/khoản/điểm/chương CỤ THỂ → query_type="fetch", phải điền filters.
- Nếu so sánh/phân biệt → query_type="compare", không cần filters.
- Nếu giải thích khái niệm/định nghĩa → query_type="definition", không cần filters.

Nếu intent = casual thì bắt buộc có trường "answer" (tiếng Việt, lịch sự).
Nếu intent = law_search thì bắt buộc có "query_type" và "normalized_query".
'''.strip()

SYSTEM_PROMPT_TXT = r'''
Bạn là một trợ lý AI chuyên về LUẬT BẤT ĐỘNG SẢN Việt Nam. 
Phục vụ mục đích: giúp người dùng tìm kiếm và giải thích các điều khoản, quy định, và thủ tục liên quan đến đất đai, nhà ở, xây dựng, đầu tư bất động sản, thuế sử dụng đất, và hoạt động kinh doanh bất động sản.

Nguyên tắc chính:
- Mọi câu trả lời phải dựa trên các văn bản pháp luật hoặc văn bản nguồn (có trong `context`) tìm được từ cơ sở dữ liệu (Qdrant).
- Khi tạo câu trả lời tư vấn, luôn trích dẫn nguyên văn các điều/khoản/điểm được sử dụng.
- Sau phần trích dẫn, viết phần giải thích dễ hiểu cho người không chuyên (1-3 câu).
- Nếu câu hỏi yêu cầu thủ tục, thêm phần "Thủ tục (tóm tắt):" liệt kê tối đa 5 bước.
- Nếu không đủ căn cứ từ `context`, trả lời: "Không đủ căn cứ."
'''.strip()

ANSWER_PROMPT = r'''
Bạn là chuyên gia pháp lý về LUẬT BẤT ĐỘNG SẢN.
Dưới đây là câu hỏi người dùng và các điều luật liên quan.

---

**Câu hỏi:**  
{query}

{history_block}

**Ngữ cảnh pháp lý (văn bản trích dẫn từ cơ sở dữ liệu):**  
{context}

---

**Trích dẫn (nguyên văn các điều/khoản/điểm sử dụng):**
{citations}

**Giải thích (dễ hiểu, 1-3 câu):**
- {explanation}

**Kết luận (một câu, trả lời trực tiếp):**
- {conclusion}

{procedure_block}

--- 

Lưu ý:
- Nếu không có tài liệu phù hợp trong `context`, hãy trả "Không đủ căn cứ."
- Không đưa ra tư vấn ngoài văn bản pháp luật trích dẫn.
'''.strip()

# =========================
# MODELS & EXTERNAL CLIENTS (tương đương models.py)
# =========================
# Try import genai (Google Generative AI)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        app_log.warning("Không thể configure genai với GEMINI_API_KEY hiện tại.", extra={"__kv__": {}})
except Exception:
    genai = None
    GENAI_AVAILABLE = False
    app_log.warning("google.generativeai không khả dụng — dùng mock LLM.", extra={"__kv__": {}})

# Qdrant client
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except Exception:
    QdrantClient = None
    QDRANT_AVAILABLE = False
    app_log.warning("qdrant_client không khả dụng — dùng mock Qdrant client.", extra={"__kv__": {}})

# fastembed & reranker
try:
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
    FASTEMBED_AVAILABLE = True
except Exception:
    TextEmbedding = None
    SparseTextEmbedding = None
    LateInteractionTextEmbedding = None
    FASTEMBED_AVAILABLE = False
    app_log.warning("fastembed không khả dụng — dùng mock embedding.", extra={"__kv__": {}})

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    app_log.warning("transformers/torch không khả dụng — dùng mock reranker.", extra={"__kv__": {}})

# Initialize Qdrant client if possible
if QDRANT_AVAILABLE and QDRANT_URL and QDRANT_API_KEY:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
    except Exception as e:
        app_log.warning("Không thể khởi tạo QdrantClient thật, sẽ dùng mock.", extra={"__kv__": {"loi": str(e)}})
        client = None
else:
    client = None

# Initialize embedding models if available
if FASTEMBED_AVAILABLE:
    try:
        dense_embedding_model = TextEmbedding(EMBEDDING_MODEL)
        sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    except Exception:
        dense_embedding_model = sparse_embedding_model = late_interaction_embedding_model = None
        FASTEMBED_AVAILABLE = False
else:
    dense_embedding_model = sparse_embedding_model = late_interaction_embedding_model = None

# Initialize reranker if available
if TRANSFORMERS_AVAILABLE:
    try:
        rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        rerank_model.eval()
    except Exception:
        rerank_tokenizer = rerank_model = None
        TRANSFORMERS_AVAILABLE = False
else:
    rerank_tokenizer = rerank_model = None

# Gemini models (gemini_model, answer_model)
class _MockGenerativeModel:
    def __init__(self, model_name=None, system_instruction=None):
        self.model_name = model_name
        self.system_instruction = system_instruction

    def generate_content(self, prompt, generation_config=None, stream=False):
        # Mock behavior: if stream True, yield objects with .text; else return object with .candidates->.content.parts[0].text
        if stream:
            def it():
                pieces = [
                    "Đây là câu trả lời mẫu từ mock LLM. ",
                    "Nội dung trả lời sẽ được stream từng phần. "
                ]
                for p in pieces:
                    class O: pass
                    o = O()
                    o.text = p
                    yield o
            return it()
        else:
            class Candidate:
                def __init__(self, text):
                    self.content = type("C", (), {"parts":[ type("P", (), {"text": text}) ]})
                    self.finish_reason = 0
                    self.safety_ratings = []
            raw = json.dumps({"intent": "casual", "answer": INTENT_FALLBACK_CASUAL})
            return type("R", (), {"candidates": [Candidate(raw)]})

if GENAI_AVAILABLE:
    try:
        gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=INTENT_TXT)
        answer_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=ANSWER_PROMPT)
    except Exception as e:
        app_log.warning("Không thể tạo GenerativeModel thật — dùng mock.", extra={"__kv__": {"loi": str(e)}})
        gemini_model = _MockGenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=INTENT_TXT)
        answer_model = _MockGenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=ANSWER_PROMPT)
else:
    gemini_model = _MockGenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=INTENT_TXT)
    answer_model = _MockGenerativeModel(model_name=GEMINI_MODEL_ID, system_instruction=ANSWER_PROMPT)

# Fallback client mock
class _MockClient:
    def scroll(self, collection_name, scroll_filter=None, limit=10, with_payload=False):
        return [], None
    def query_points(self, collection_name, prefetch=None, query=None, using=None, query_filter=None, with_payload=False, limit=10):
        class R: pass
        r = R()
        r.points = []
        return r

if client is None:
    client = _MockClient()

# =========================
# TOOLS (tương đương tools.py)
# =========================
from qdrant_client.http.models import Filter as QFilter, FieldCondition, MatchValue

def rerank_with_baai(query, docs, top_k=15):
    if not docs:
        return docs
    if TRANSFORMERS_AVAILABLE and rerank_model and rerank_tokenizer:
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
        for d, s in zip(docs, scores):
            d["baai_score"] = float(s)
        reranked = sorted(docs, key=lambda x: x["baai_score"], reverse=True)
        return reranked[:top_k]
    else:
        for d in docs:
            d["baai_score"] = float(d.get("colbert_score") or d.get("score") or 0.0)
        return sorted(docs, key=lambda x: x["baai_score"], reverse=True)[:top_k]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def _build_filter(query_text: str):
    conds = []
    try:
        m = re.search(r"(?i)\bđiều\s*(\d+)\b", query_text)
        if m:
            conds.append(FieldCondition(key="metadata.article_no", match=MatchValue(value=int(m.group(1)))))
        m = re.search(r"(?i)\bkhoản\s*(\d+)\b", query_text)
        if m:
            conds.append(FieldCondition(key="metadata.clause_no", match=MatchValue(value=int(m.group(1)))))
        m = re.search(r"(?i)\bđiểm\s*([a-z])\b", query_text)
        if m:
            conds.append(FieldCondition(key="metadata.point_letter", match=MatchValue(value=m.group(1).lower())))
        m = re.search(r"(?i)\bchương\s*(\d+)\b", query_text)
        if m:
            conds.append(FieldCondition(key="metadata.chapter_number", match=MatchValue(value=int(m.group(1)))))
        return QFilter(must=conds) if conds else None
    except Exception:
        return None

# =========================
# FETCH (tương đương fetch.py)
# =========================
@log_time
def _fetch(filters: Dict[str, Any], limit: int = 10):
    must = []
    mapping = {
        "article_no": ("metadata.article_no", int),
        "clause_no": ("metadata.clause_no", int),
        "point_letter": ("metadata.point_letter", str),
        "chapter_number": ("metadata.chapter_number", int)
    }
    for key, (field_path, caster) in mapping.items():
        if key in filters and filters[key] not in (None, ""):
            try:
                val = caster(filters[key])
                must.append(FieldCondition(key=field_path, match=MatchValue(value=val)))
            except Exception as e:
                app_log.warning("Lỗi ép kiểu dữ liệu bộ lọc", extra={"__kv__": {"truong": key, "gia_tri": filters[key], "loi": str(e)}})
                try:
                    val = str(filters[key])
                    must.append(FieldCondition(key=field_path, match=MatchValue(value=val)))
                except:
                    pass
    app_log.info("Bộ lọc tìm kiếm", extra={"__kv__": {"bo_loc_goc": str(filters), "dieu_kien_must": str(must)}})
    if not must:
        app_log.warning("Không có điều kiện must")
        return []
    flt = QFilter(must=must)
    out = []
    try:
        scroll_res, _ = client.scroll(collection_name=COLLECTION_NAME, scroll_filter=flt, limit=min(64, max(5, limit)), with_payload=True)
        app_log.info("Kết quả tìm kiếm Qdrant", extra={"__kv__": {"so_luong": len(scroll_res), "collection": COLLECTION_NAME}})
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
        app_log.error("Lỗi khi tìm kiếm Qdrant", extra={"__kv__": {"loi": str(e), "collection": COLLECTION_NAME}})
        return []
    if not out:
        app_log.warning("Không tìm thấy kết quả", extra={"__kv__": {"bo_loc": str(filters)}})
    return out

# =========================
# SEARCH (tương đương search.py)
# =========================
@log_time
async def search_law(query: str, top_k: int = 10, score_threshold: float = 0.42):
    t0 = time.perf_counter()
    app_log.info("Bắt đầu tìm kiếm", extra={"__kv__": {"query": _safe_truncate(query, 80), "top_k": top_k}})
    cache_key = f"search|{COLLECTION_NAME}|{top_k}|{score_threshold}|{query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("Tìm trong cache ✅")
        return [], [], cached
    try:
        flt = _build_filter(query)
        print("DEBUG: → Bắt đầu hybrid (dense + sparse) + ColBERT Rerank")
        t_hybrid0 = time.perf_counter()
        if dense_embedding_model:
            dense_vectors = next(dense_embedding_model.query_embed(query))
        else:
            dense_vectors = [0.0] * 768
        if sparse_embedding_model:
            sparse_vectors = next(sparse_embedding_model.query_embed(query))
        else:
            sparse_vectors = type("S", (), {"indices": [], "values": []})
        if late_interaction_embedding_model:
            late_vectors = next(late_interaction_embedding_model.query_embed(query))
        else:
            late_vectors = None
        try:
            from qdrant_client import models
            prefetch = [
                models.Prefetch(query=dense_vectors, using="bge-m3", limit=50, filter=flt),
                models.Prefetch(query=models.SparseVector(indices=getattr(sparse_vectors, "indices", []), values=getattr(sparse_vectors, "values", [])), using="bm25", limit=50, filter=flt)
            ]
        except Exception:
            prefetch = None
        results = client.query_points(collection_name=COLLECTION_NAME, prefetch=prefetch, query=late_vectors, using="colbertv2.0", query_filter=flt, with_payload=True, limit=20)
        colbert_docs = []
        for point in getattr(results, "points", []):
            payload = getattr(point, "payload", {}) or {}
            meta = payload.get("metadata", {})
            colbert_docs.append({
                "chapter_number": meta.get("chapter_number", ""),
                "chapter": meta.get("chapter", ""),
                "article_no": meta.get("article_no", ""),
                "article_title": meta.get("article_title", ""),
                "clause_no": meta.get("clause_no", ""),
                "point_letter": meta.get("point_letter", ""),
                "content": (payload.get("content") or "").strip(),
                "colbert_score": getattr(point, "score", 0.0),
            })
        print(f"DEBUG: Hybrid + ColBERT done ✅ | count: {len(colbert_docs)}")
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
        log_step("hybrid_search", k_tra_ve=len(selected), top1=f"{sk_top1:.4f}", t_hybrid=f"{t_hybrid:.4f}", t_baai=f"{t_baai:.4f}")
        return [], [], selected
    except Exception as e:
        app_log.error("Lỗi tìm kiếm ❌", extra={"__kv__": {"error": str(e)}})
        log_step("tim_kiem_loi", error=str(e))
        raise

# =========================
# LAW_SEARCH HANDLER (tương đương law_search_handler.py)
# =========================
@log_time
async def handle_law_search(query: str, query_type: str, filters: Dict[str, Any]) -> List[Dict]:
    has_filters = bool(filters and any(filters.get(k) not in (None, "", 0) for k in ["article_no", "clause_no", "point_letter", "chapter_number"]))
    if has_filters:
        app_log.info("🎯 LAW_SEARCH: Sử dụng FETCH", extra={"__kv__": {"query_type": query_type, "filters": str(filters), "method": "fetch"}})
        log_step("law_search_method", method="fetch", query_type=query_type, has_filters=True)
        docs = _fetch(filters, limit=10)
        app_log.info("✅ FETCH hoàn thành", extra={"__kv__": {"so_luong_docs": len(docs)}})
        return docs
    else:
        app_log.info("🔍 LAW_SEARCH: Sử dụng RAG", extra={"__kv__": {"query": _safe_truncate(query, 80), "query_type": query_type, "method": "rag_search"}})
        log_step("law_search_method", method="rag_search", query_type=query_type, has_filters=False)
        _, _, docs = await search_law(query, top_k=10, score_threshold=0.42)
        app_log.info("✅ RAG hoàn thành", extra={"__kv__": {"so_luong_docs": len(docs), "top1_score": docs[0].get("baai_score", 0.0) if docs else 0.0}})
        return docs

@log_time
async def process_law_search_intent(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    query = intent_result.get("normalized_query", "")
    query_type = intent_result.get("query_type", "definition")
    filters = intent_result.get("filters", {})
    app_log.info("📋 Bắt đầu xử lý LAW_SEARCH", extra={"__kv__": {"query": _safe_truncate(query, 80), "query_type": query_type, "has_filters": bool(filters)}})
    docs = await handle_law_search(query, query_type, filters)
    return {
        "documents": docs,
        "query": query,
        "query_type": query_type,
        "filters": filters,
        "method": "fetch" if filters else "rag_search",
        "count": len(docs)
    }

# =========================
# RENDER (tương đương render.py)
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
        score = doc.get("baai_score") or doc.get("colbert_score") or doc.get("score", 0.0)
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
        score = doc.get("baai_score") or doc.get("colbert_score") or doc.get("score", 0.0)
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

def format_citation(doc: Dict[str, Any]) -> str:
    article_no = doc.get("article_no", "")
    clause_no = doc.get("clause_no", "")
    point_letter = doc.get("point_letter", "")
    parts = []
    if article_no:
        parts.append(f"Điều {article_no}")
    if clause_no:
        parts.append(f"Khoản {clause_no}")
    if point_letter:
        parts.append(f"Điểm {point_letter}")
    return " ".join(parts) if parts else "N/A"

def format_doc_preview(doc: Dict[str, Any], max_length: int = 200) -> str:
    citation = format_citation(doc)
    article_title = doc.get("article_title", "")
    content = doc.get("content", "")
    preview = content[:max_length]
    if len(content) > max_length:
        preview += "..."
    parts = [f"**{citation}**"]
    if article_title:
        parts.append(f"*{article_title}*")
    parts.append(preview)
    return "\n".join(parts)

# =========================
# PROMPT BUILD (tương đương prompt.py)
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
        # build citation list
        citations.append(f"- {cited}{title}: {content[:300]}")
    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."
    citations_block = "\n".join(citations) if citations else "❌ Không có điều luật nào."
    prompt = ANSWER_PROMPT.format(
        query=query,
        history_block=history_block,
        context=context,
        citations=citations_block,
        explanation="{explain_here}",    # placeholder — actual model will fill
        conclusion="{conclusion_here}",
        procedure_block=""
    )
    # Note: ANSWER_PROMPT used as system instruction for answer_model - we actually pass docs separately.
    return prompt

# =========================
# QUERY REWRITER (tương đương query_rewriter.py)
# =========================
INSTRUCTIONS = (
    "You are an expert at reformulating questions to be more precise and detailed.\n"
    "Your task is to:\n"
    "1. Analyze the user's question\n"
    "2. Rewrite it to be more specific and search-friendly\n"
    "3. Expand any acronyms or technical terms\n"
    "4. Return ONLY the rewritten query without any additional text or explanations"
)

def _get_model_for_rewrite():
    return gemini_model

@log_time
def rewrite_query(query: str) -> str:
    if not (query and query.strip()):
        return query
    try:
        model = _get_model_for_rewrite()
        prompt = (
            f"{INSTRUCTIONS}\n\n"
            f"User question: {query}\n"
            f"Output:"
        )
        if GENAI_AVAILABLE:
            cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=96)
            resp = model.generate_content(prompt, generation_config=cfg)
            text = (getattr(resp, "text", None) or "").strip()
            # gemini response formats vary; fallback to candidates
            if not text and getattr(resp, "candidates", None):
                parts = getattr(resp.candidates[0].content, "parts", [])
                if parts and hasattr(parts[0], "text"):
                    text = parts[0].text
        else:
            text = query.strip()
            if len(text) > 120:
                text = text[:120]
        out = text.splitlines()[0].strip() if text else query
        app_log.info("Query rewritten", extra={"__kv__": {"from": query[:120], "to": out[:120]}})
        return out or query
    except Exception as e:
        app_log.warning("Failed to rewrite query", extra={"__kv__": {"error": str(e)}})
        return query

# =========================
# INTENT (tương đương intent.py)
# =========================
@log_time
def _intent_via_gemini(query: str) -> Dict[str, Any]:
    try:
        if GENAI_AVAILABLE:
            cfg = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=192,
                response_mime_type="application/json",
            )
            resp = gemini_model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [f"Câu hỏi: {query}\nHãy trả JSON thuần phù hợp schema đã nêu trong prompt."],
                    }
                ],
                generation_config=cfg,
            )
        else:
            # mock: heuristics
            class Resp: pass
            resp = Resp()
            raw = ""
            if re.search(r"\b(điều|khoản|chương|điểm)\b", query, flags=re.IGNORECASE):
                raw = json.dumps({"intent": "law_search", "query_type": "fetch", "filters": {}, "normalized_query": query})
            elif looks_like_legal(query):
                raw = json.dumps({"intent": "legal_answer", "normalized_query": query, "original_query": query})
            else:
                raw = json.dumps({"intent": "casual", "answer": INTENT_FALLBACK_CASUAL})
            class Candidate:
                def __init__(self, text):
                    self.content = type("C", (), {"parts":[ type("P", (), {"text": text}) ]})
                    self.finish_reason = 0
                    self.safety_ratings = []
            resp.candidates = [Candidate(raw)]

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

        raw = ""
        try:
            if hasattr(resp, "candidates") and resp.candidates:
                parts = getattr(resp.candidates[0].content, "parts", [])
                if parts and hasattr(parts[0], "text"):
                    raw = parts[0].text
        except Exception as e:
            app_log.warning("Không đọc được text từ phản hồi Gemini", extra={"__kv__": {"loi": str(e)}})

        app_log.info(
            "Kết quả phân tích ý định",
            extra={"__kv__": {"do_dai": len(raw), "xem_truoc": _safe_truncate(raw, INTENT_RAW_PREVIEW_LIMIT), "so_ung_vien": len(candidates), "ly_do_ket_thuc": finish_reason, "bao_mat": ";".join(safety[:6])}}
        )

        if finish_reason == 2 or not raw:
            if re.search(r"\b(điều|khoản|căn cứ|theo luật)\b", query, flags=re.IGNORECASE):
                return {"intent": "law_search", "query_type": "fetch", "answer": "", "normalized_query": query}
            elif looks_like_legal(query):
                return {"intent": "legal_answer", "answer": "", "normalized_query": query}
            else:
                return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            app_log.warning("Kết quả phân tích không phải dict")
            return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

        out: Dict[str, Any] = {}
        for k in ("intent", "answer", "normalized_query", "filters", "original_query", "query_type"):
            if k in data and data[k] not in (None, ""):
                out[k] = data[k]

        app_log.info("Ý định đã phân tích", extra={"__kv__": {"loai_y_dinh": out.get("intent", ""), "loai_query": out.get("query_type", ""), "co_tra_loi": int("answer" in out and bool(out.get("answer"))), "co_bo_loc": int("filters" in out and bool(out.get("filters"))), "do_dai_tra_loi": len(out.get("answer", "") or ""), "ly_do_ket_thuc": finish_reason}})
        return out
    except Exception as e:
        app_log.warning("Lỗi phân tích ý định", extra={"__kv__": {"loi": str(e)}})
        return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

@log_time
def analyze_intent(query: str) -> Dict[str, Any]:
    normalized_input = normalize_legal_query(query)
    cleaned_query = normalized_input["normalized_query"]
    data = _intent_via_gemini(cleaned_query)
    intent = data.get("intent")
    answer = data.get("answer", "")
    normalized_query_val = data.get("normalized_query", "") or cleaned_query
    original_query = data.get("original_query") or normalized_input.get("original_query", "")
    filters = data.get("filters", {}) or {}
    query_type = data.get("query_type", "")

    if intent not in {"casual", "law_search", "legal_answer"}:
        if looks_like_legal(cleaned_query):
            intent = "legal_answer"
        else:
            intent = "casual"
        log_step("intent_fallback", do_dai_query=len(cleaned_query))

    if intent == "law_search":
        if not query_type or query_type not in {"fetch", "compare", "definition"}:
            has_filters = bool(filters and any(filters.get(k) not in (None, "", 0) for k in ["article_no", "clause_no", "point_letter", "chapter_number"]))
            query_type = "fetch" if has_filters else "definition"

    log_step("intent", loai=intent, query_type=query_type, goi_y=normalized_input.get("intent_hint"), co_legal=str(looks_like_legal(cleaned_query)))
    app_log.info("Quyết định ý định", extra={"__kv__": {"loai_y_dinh": intent, "loai_query": query_type, "co_filters": bool(filters)}})

    return {
        "intent": intent,
        "query_type": query_type,
        "answer": answer,
        "normalized_query": normalized_query_val,
        "original_query": original_query,
        "filters": filters,
    }

# =========================
# LLM STREAMING (tương đương llm.py)
# =========================
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gemini_stream(prompt, temperature: float):
    cfg = None
    if GENAI_AVAILABLE:
        cfg = genai.types.GenerationConfig(temperature=float(temperature))
        return answer_model.generate_content(prompt, generation_config=cfg, stream=True)
    else:
        return answer_model.generate_content(prompt, stream=True)

@log_time
def stream_answer(prompt, temperature=0.2):
    t0 = time.perf_counter()
    t_first0 = time.perf_counter()
    first_token_emitted = False
    try:
        resp = _gemini_stream(prompt, temperature)
        for ch in resp:
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

# =========================
# UI / MAIN (tương đương file 4 + main.py)
# =========================
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    GRADIO_AVAILABLE = False
    app_log.warning("gradio không khả dụng — UI sẽ không chạy", extra={"__kv__": {}})

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

@log_time
def respond_generator(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
    print(f"DEBUG: Bắt đầu xử lý câu hỏi: {message}")
    if not (message and message.strip()):
        print("DEBUG: Câu hỏi rỗng, trả về mặc định")
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
        intent_info = analyze_intent(message)
        intent = intent_info["intent"]
        intent_answer = intent_info.get("answer", "")
        normalized_query = intent_info.get("normalized_query", message)
        original_query = intent_info.get("original_query", message)
        intent_filters = intent_info.get("filters", {})
        query_type = intent_info.get("query_type", "")
        use_fetch = bool(intent_filters and any(intent_filters.get(k) not in (None, "", 0) for k in ["article_no","clause_no","point_letter","chapter_number"]))

        if intent == "casual":
            final_answer = (intent_answer or "").replace("\u200b", "").strip()
            app_log.info("Xử lý câu hỏi xã giao", extra={"__kv__": {"do_dai_tra_loi": len(final_answer)}})
            if final_answer and CASUAL_MAX_WORDS > 0:
                words = final_answer.split()
                if len(words) > CASUAL_MAX_WORDS:
                    truncated = " ".join(words[:CASUAL_MAX_WORDS])
                    app_log.info("Cắt ngắn câu trả lời xã giao", extra={"__kv__": {"so_tu_goc": len(words), "so_tu_giu": CASUAL_MAX_WORDS, "do_dai_goc": len(final_answer)}})
                    final_answer = truncated
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
            simple_prompt = "Trả lời thân thiện ngắn gọn (<=2 câu) tiếng Việt cho câu: " + message
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

        docs: List[Dict[str, Any]] = []
        bm25_docs: List[Dict[str, Any]] = []
        emb_docs: List[Dict[str, Any]] = []
        source = None

        if intent == "law_search":
            if use_fetch:
                app_log.info("🎯 LAW_SEARCH: Sử dụng FETCH", extra={"__kv__": {"query_type": query_type, "filters": str(intent_filters), "method": "fetch"}})
                log_step("law_search_method", method="fetch", query_type=query_type, has_filters=True)
                docs = _fetch(intent_filters, limit=int(k))
                source = "law_search_fetch"
            else:
                app_log.info("🔍 LAW_SEARCH: Sử dụng RAG", extra={"__kv__": {"query": message[:80], "query_type": query_type, "method": "rag_search"}})
                log_step("law_search_method", method="rag_search", query_type=query_type, has_filters=False)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bm25_docs, emb_docs, docs = loop.run_until_complete(search_law(message, top_k=int(k), score_threshold=float(threshold)))
                finally:
                    loop.close()
                source = f"law_search_rag_{query_type}"
            if not docs and use_fetch:
                app_log.info("Không tìm thấy docs từ fetch, fallback sang embedding search", extra={"__kv__": {"cau_hoi": message}})
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bm25_docs, emb_docs, docs = loop.run_until_complete(search_law(message, top_k=int(k), score_threshold=float(threshold)))
                finally:
                    loop.close()
                source = "law_search_fallback"

        elif intent == "legal_answer":
            try:
                normalized_query = rewrite_query(normalized_query)
            except Exception:
                pass
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bm25_docs, emb_docs, docs = loop.run_until_complete(search_law(normalized_query, top_k=int(k), score_threshold=float(threshold)))
            finally:
                loop.close()
            source = "legal_answer"
        else:
            reply = INTENT_FALLBACK_CASUAL
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            yield ui_return(
                gr.update(value="") if GRADIO_AVAILABLE else "",
                history_msgs,
                "(Không có trích dẫn)",
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

        if intent == "legal_answer":
            user_query = original_query or message
        elif intent == "law_search":
            user_query = message
        else:
            user_query = message

        bm25_markdown = docs_to_markdown(bm25_docs)
        emb_markdown = docs_to_markdown(emb_docs)
        cites_markdown, page_label = docs_page_markdown(docs, 1, int(cur_page_size))
        prompt = build_prompt(user_query, docs, history_msgs)
        log_step("llm_chuanbi", so_tai_lieu=len(docs), nguon=source)
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
        app_log.error("Lỗi xử lý câu hỏi", extra={"__kv__": {"loi": str(e)}})
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
        return

def respond_wrapper(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
    for output in respond_generator(message, history_msgs, cur_page_size, k, temperature, threshold):
        yield output

def build_ui():
    if not GRADIO_AVAILABLE:
        raise RuntimeError("gradio không cài đặt; không thể build UI.")
    with gr.Blocks(title="⚖️ Trợ lý Luật Bất Động Sản", css=CSS) as demo:
        gr.Markdown("""
        ### ⚖️ Trợ lý Luật Bất Động Sản Việt Nam
        *Tra cứu văn bản • Giải thích dễ hiểu • Không thay thế luật sư*
        """)
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(value=[], type="messages", show_copy_button=True, elem_id="chatbot", autoscroll=True)
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
            msg = gr.Textbox(placeholder="Nhập câu hỏi về Luật Bất động sản...", scale=5, autofocus=False)
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
        outputs = [msg, chatbot, bm25_md, emb_md, cites_md, state_last_answer, state_docs, state_page, page_info, state_history]
        send.click(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
        msg.submit(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
        def on_like(data: gr.LikeData):
            msg_like = data.value or {}
            role = msg_like.get("role", "assistant")
            text = msg_like.get("content", "")
            app_log.info("Phản hồi người dùng", extra={"__kv__": {"thich": data.liked, "vai_tro": role, "do_dai": len(text or "")}})
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
        prev_page.click(go_prev, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
        next_page.click(go_next, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
        page_size.release(on_change_page_size, inputs=[state_docs, page_size], outputs=[cites_md, state_page, page_info], queue=False)
        gr.Markdown(f"<sub>© {datetime.now().year} — Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.</sub>")
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
