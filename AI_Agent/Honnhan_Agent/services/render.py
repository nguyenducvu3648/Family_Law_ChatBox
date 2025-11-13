"""
Module xử lý render documents thành markdown để hiển thị trong UI.
Đã cập nhật để xử lý cấu trúc metadata mới.
"""
from typing import List, Dict, Any, Tuple


def docs_to_markdown(docs: List[Dict[str, Any]]) -> str:
    """
    Chuyển đổi danh sách documents thành markdown.
    
    Args:
        docs: List các documents với structure mới (metadata nested)
    
    Returns:
        String markdown
    """
    if not docs:
        return "(Chưa có dữ liệu)"
    
    lines = []
    for i, doc in enumerate(docs, 1):
        article_no = doc.get("article_no", "")
        article_title = doc.get("article_title", "")
        clause_no = doc.get("clause_no", "")
        point_letter = doc.get("point_letter", "")
        content = doc.get("content", "")[:150]  # Preview 150 chars
        score = doc.get("baai_score") or doc.get("colbert_score") or doc.get("score", 0.0)
        
        # Build citation
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
        lines.append("")  # Empty line
    
    return "\n".join(lines)


def paginate_docs(docs: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[List[Dict], int, int, int]:
    """
    Phân trang documents.
    
    Args:
        docs: List documents
        page: Số trang hiện tại (1-indexed)
        page_size: Số items mỗi trang
    
    Returns:
        (paginated_docs, total_docs, total_pages, current_page)
    """
    if not docs:
        return [], 0, 0, 1
    
    total = len(docs)
    total_pages = (total + page_size - 1) // page_size  # Ceiling division
    current_page = max(1, min(page, total_pages))
    
    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated = docs[start_idx:end_idx]
    
    return paginated, total, total_pages, current_page


def docs_page_markdown(docs: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[str, str]:
    """
    Chuyển đổi một trang documents thành markdown.
    
    Args:
        docs: List tất cả documents
        page: Số trang hiện tại
        page_size: Số items mỗi trang
    
    Returns:
        (markdown_string, page_label)
    """
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
        
        # Build citation
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
    """
    Format citation từ một document.
    
    Args:
        doc: Document dict
    
    Returns:
        Citation string (ví dụ: "Điều 10 Khoản 2 Điểm a")
    """
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
    """
    Format document preview với citation và content.
    
    Args:
        doc: Document dict
        max_length: Độ dài tối đa của content preview
    
    Returns:
        Formatted string
    """
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