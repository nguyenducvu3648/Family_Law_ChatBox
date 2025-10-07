from textwrap import dedent
from typing import List, Dict, Any

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