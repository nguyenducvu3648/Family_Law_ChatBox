import json
import re
from typing import Dict, Any
import google.generativeai as genai
from core.logging_setup import app_log, log_step, log_time
from models.models import gemini_model
from core.config import INTENT_FALLBACK_CASUAL, INTENT_RAW_PREVIEW_LIMIT
from utils.utils import _safe_truncate, looks_like_legal, normalize_legal_query

@log_time
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
        
        app_log.info(
            "Ý định đã phân tích",
            extra={
                "__kv__": {
                    "loai_y_dinh": out.get("intent", ""),
                    "loai_query": out.get("query_type", ""),
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

@log_time
def analyze_intent(query: str) -> Dict[str, Any]:
    """
    Phân tích intent của câu hỏi.
    
    Returns:
        Dict với các key:
        - intent: "casual" | "legal_answer" | "law_search"
        - query_type: (nếu law_search) "fetch" | "compare" | "definition"
        - answer: (nếu casual) câu trả lời trực tiếp
        - normalized_query: câu hỏi đã chuẩn hóa
        - original_query: câu hỏi gốc
        - filters: (nếu law_search và query_type="fetch") dict các filters
    """
    normalized_input = normalize_legal_query(query)
    cleaned_query = normalized_input["normalized_query"]

    data = _intent_via_gemini(cleaned_query)
    intent = data.get("intent")
    answer = data.get("answer", "")
    normalized_query = data.get("normalized_query", "") or cleaned_query
    original_query = data.get("original_query") or normalized_input.get("original_query", "")
    filters = data.get("filters", {}) or {}
    query_type = data.get("query_type", "")

    # Fallback logic nếu intent không hợp lệ
    if intent not in {"casual", "law_search", "legal_answer"}:
        if looks_like_legal(cleaned_query):
            intent = "legal_answer"
        else:
            intent = "casual"
        log_step("intent_fallback", do_dai_query=len(cleaned_query))

    # Đảm bảo law_search luôn có query_type hợp lệ
    if intent == "law_search":
        if not query_type or query_type not in {"fetch", "compare", "definition"}:
            # Auto-detect dựa vào filters
            has_filters = bool(filters and any(
                filters.get(k) not in (None, "", 0) 
                for k in ["article_no", "clause_no", "point_letter", "chapter_number"]
            ))
            query_type = "fetch" if has_filters else "definition"

    log_step(
        "intent", 
        loai=intent, 
        query_type=query_type,
        goi_y=normalized_input.get("intent_hint"), 
        co_legal=str(looks_like_legal(cleaned_query))
    )
    
    app_log.info(
        "Quyết định ý định", 
        extra={
            "__kv__": {
                "loai_y_dinh": intent,
                "loai_query": query_type,
                "co_filters": bool(filters)
            }
        }
    )
    
    return {
        "intent": intent,
        "query_type": query_type,
        "answer": answer,
        "normalized_query": normalized_query,
        "original_query": original_query,
        "filters": filters,
    }