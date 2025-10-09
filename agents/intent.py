import json
import re
from typing import Dict, Any
import google.generativeai as genai
from core.logging_setup import app_log, log_step, log_time
from models.models import gemini_model
from core.config import INTENT_FALLBACK_CASUAL, INTENT_RAW_PREVIEW_LIMIT
from utils.utils import _safe_truncate, looks_like_legal

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

@log_time
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