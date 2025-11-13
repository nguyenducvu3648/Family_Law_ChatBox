"""
Handler xử lý law_search với logic:
- Có filters → gọi fetch (nhanh)
- Không có filters → gọi RAG search_law (function calling style)
"""
from typing import List, Dict, Any
from core.logging_setup import app_log, log_step, log_time
from retrieval.fetch import _fetch
from retrieval.search import search_law
from utils.utils import _safe_truncate


@log_time
async def handle_law_search(
    query: str,
    query_type: str,
    filters: Dict[str, Any]
) -> List[Dict]:
    """
    Xử lý law_search với logic fetch vs RAG.
    
    Args:
        query: Câu hỏi đã normalize
        query_type: "fetch" | "compare" | "definition"
        filters: Dict chứa article_no, clause_no, point_letter, chapter_number
    
    Returns:
        List các documents
    """
    # Kiểm tra xem có filters hợp lệ không
    has_filters = bool(filters and any(
        filters.get(k) not in (None, "", 0) 
        for k in ["article_no", "clause_no", "point_letter", "chapter_number"]
    ))
    
    if has_filters:
        # CÓ FILTERS → Dùng FETCH (nhanh, chính xác)
        app_log.info(
            "🎯 LAW_SEARCH: Sử dụng FETCH",
            extra={"__kv__": {
                "query_type": query_type,
                "filters": str(filters),
                "method": "fetch"
            }}
        )
        log_step("law_search_method", method="fetch", query_type=query_type, has_filters=True)
        
        docs = _fetch(filters, limit=10)
        
        app_log.info(
            "✅ FETCH hoàn thành",
            extra={"__kv__": {"so_luong_docs": len(docs)}}
        )
        return docs
    
    else:
        # KHÔNG CÓ FILTERS → Dùng RAG (function calling style)
        app_log.info(
            "🔍 LAW_SEARCH: Sử dụng RAG",
            extra={"__kv__": {
                "query": _safe_truncate(query, 80),
                "query_type": query_type,
                "method": "rag_search"
            }}
        )
        log_step("law_search_method", method="rag_search", query_type=query_type, has_filters=False)
        
        # Gọi RAG search_law (function calling)
        # search_law trả về: (dense_results, sparse_results, hybrid_results)
        _, _, docs = await search_law(query, top_k=10, score_threshold=0.42)
        
        app_log.info(
            "✅ RAG hoàn thành",
            extra={"__kv__": {
                "so_luong_docs": len(docs),
                "top1_score": docs[0].get("baai_score", 0.0) if docs else 0.0
            }}
        )
        return docs


@log_time
async def process_law_search_intent(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý toàn bộ flow của law_search intent.
    
    Args:
        intent_result: Kết quả từ analyze_intent()
    
    Returns:
        Dict chứa documents và metadata
    """
    query = intent_result.get("normalized_query", "")
    query_type = intent_result.get("query_type", "definition")
    filters = intent_result.get("filters", {})
    
    app_log.info(
        "📋 Bắt đầu xử lý LAW_SEARCH",
        extra={"__kv__": {
            "query": _safe_truncate(query, 80),
            "query_type": query_type,
            "has_filters": bool(filters)
        }}
    )
    
    # Gọi handler chính
    docs = await handle_law_search(query, query_type, filters)
    
    return {
        "documents": docs,
        "query": query,
        "query_type": query_type,
        "filters": filters,
        "method": "fetch" if filters else "rag_search",
        "count": len(docs)
    }