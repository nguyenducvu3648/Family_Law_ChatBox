from typing import List, Dict, Any
import re
import os
# Import hàm từ prompt_loader (đặt file này ở cùng cấp hoặc trong PYTHONPATH)
from core.prompt_loader import load_prompt

# Xác định thư mục gốc của agent
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# -> BASE_DIR = Family-law-chatbot/AI_Agent/Honnhan_Agent

# Gọi load_prompt với folder và loại prompt
ANSWER_PROMPT = load_prompt(BASE_DIR, "answer")

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

    # Format vào template đã load
    prompt = ANSWER_PROMPT.format(
        query=query,
        history_block=history_block,
        context=context
    )

    return prompt