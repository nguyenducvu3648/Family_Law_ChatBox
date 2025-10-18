from typing import Any, Dict, List, Tuple 

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
        lines.append(
            f"**{i}. {cited}{chapter}{title}**  \n"
            f"{content}  \n"
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