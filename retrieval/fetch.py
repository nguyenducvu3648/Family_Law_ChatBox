from typing import Dict, Any

from core.logging_setup import app_log, log_time
from models.models import client
from core.config import COLLECTION_NAME
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

@log_time
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