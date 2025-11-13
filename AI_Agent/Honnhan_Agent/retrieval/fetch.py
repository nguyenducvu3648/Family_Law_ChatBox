from typing import Dict, Any

from core.logging_setup import app_log, log_time
from models.models import client
from core.config import COLLECTION_NAME
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

@log_time
def _fetch(filters: Dict[str, Any], limit: int = 10):
    must = []
    # Mapping với nested path trong metadata
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
                app_log.warning(
                    "Lỗi ép kiểu dữ liệu bộ lọc",
                    extra={"__kv__": {"truong": key, "gia_tri": filters[key], "loi": str(e)}},
                )
                try:
                    val = str(filters[key])
                    must.append(FieldCondition(key=field_path, match=MatchValue(value=val)))
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
            # Lấy metadata
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
        app_log.error(
            "Lỗi khi tìm kiếm Qdrant",
            extra={"__kv__": {"loi": str(e), "collection": COLLECTION_NAME}},
        )
        return []
    
    if not out:
        app_log.warning("Không tìm thấy kết quả", extra={"__kv__": {"bo_loc": str(filters)}})
    
    return out