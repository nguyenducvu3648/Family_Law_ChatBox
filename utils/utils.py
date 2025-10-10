import re

from typing import Any, Dict, List

LEGAL_HINTS = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|hôn nhân|ly hôn|ly thân|nuôi con|tài sản|"
    r"quan hệ vợ chồng|kết hôn|hủy kết hôn|chung sống như vợ chồng|cấp dưỡng|giám hộ)\b"
)

def looks_like_legal(query: str) -> bool:
    return bool(LEGAL_HINTS.search(query or ""))

def _safe_truncate(text: str, limit: int = 800) -> str:
    return text if text and len(text) <= limit else (text[:limit] + "…(cắt)") if text else ""

def normalize_legal_query(query: str) -> dict:
    """
    Chuẩn hóa câu hỏi pháp lý để mô hình hiểu đúng ý định và ngữ nghĩa.
    Xử lý các dạng câu:
    - Nhận định / đúng sai
    - Hỏi quy định pháp luật
    - Tư vấn / tình huống cụ thể
    - Giải thích khái niệm
    - Hỏi mức phạt / chế tài
    Đồng thời: sửa lỗi chính tả, thêm dấu hỏi hợp ngữ cảnh, gom câu.
    """

    original = (query or "").strip()
    text = re.sub(r"\s+", " ", original).strip()
    text = text[0].upper() + text[1:] if text else text

    # --- 1. Sửa lỗi chính tả phổ biến trong ngữ cảnh luật ---
    corrections = {
        "nghĩ vụ": "nghĩa vụ",
        "cấp dưởng": "cấp dưỡng",
        "nuôi dưởng": "nuôi dưỡng",
        "pháp luât": "pháp luật",
        "hôn nhân gia đình": "hôn nhân và gia đình",
        "truy tố hình sự": "truy cứu trách nhiệm hình sự",
        "được phép không": "có được không",
        "được": "được",
        "phai": "phải",
        "co": "có",
    }
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)

    # --- 2. Gộp nhiều câu rời thành một ý duy nhất ---
    text = re.sub(r"\s*[.!]+\s*", ". ", text)
    text = text.strip().rstrip(".").strip()

    # --- 3. Phân loại sơ bộ ý định ---
    intent = "general_legal"  # mặc định

    # --- Ưu tiên cao nhất: So sánh / phân biệt ---
    if re.search(r"\b(phân biệt|so sánh|khác nhau|điểm giống|điểm khác|giữa)\b", text, flags=re.IGNORECASE):
        intent = "compare"

    # --- Nhận định / Đúng sai ---
    elif re.search(r"\b(phải|có|được|bị|nên)\b", text, flags=re.IGNORECASE):
        intent = "true_false"

    # --- Tình huống / tư vấn / nêu ý kiến ---
    elif re.search(
        r"\b(nếu|trường hợp|giả sử|muốn hỏi|muốn biết|nên|có nên|làm sao|làm thế nào|cách nào|xử lý ra sao|xử lý thế nào|khởi kiện|hòa giải|phải làm gì)\b",
        text,
        flags=re.IGNORECASE,
    ):
        intent = "advice"

    # --- Giải thích khái niệm ---
    elif re.search(r"\b(là gì|được hiểu như thế nào|định nghĩa)\b", text, flags=re.IGNORECASE):
        intent = "definition"

    # --- Mức phạt / chế tài ---
    elif re.search(r"\b(mức phạt|xử phạt|phạt tiền|chế tài)\b", text, flags=re.IGNORECASE):
        intent = "punishment"

    # --- Hỏi quy định / viện dẫn luật ---
    elif re.search(r"\b(theo luật|theo quy định|căn cứ)\b", text, flags=re.IGNORECASE):
        intent = "law_reference"

    # --- Dự phòng: câu mô tả có “là / thuộc / được coi là” mà chưa có dấu hỏi ---
    elif re.search(
        r"\b(là|thuộc|bao gồm|gồm|được coi là|được xem là|được xác định là|có nghĩa là|được tính là|phải|được quyền|có nghĩa vụ|chịu trách nhiệm)\b",
        text,
        flags=re.IGNORECASE,
    ):
        intent = "true_false"
        if not text.endswith("?"):
            text = text.rstrip(".") + "?"


    # --- 3.5. Nếu là câu nhận định kiểu "A là B" (ví dụ: "Tài sản ... là tài sản ...") ---
    if intent == "general_legal":
        if re.search(r"\b(là|thuộc|bao gồm|gồm|được coi là|được xem là|được xác định là|có nghĩa là|được tính là|phải|được quyền|có nghĩa vụ|chịu trách nhiệm)\b", text, flags=re.IGNORECASE):
            intent = "true_false"
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"

    # --- 4. Thêm dấu hỏi hợp ngữ cảnh ---
    if not text.endswith("?"):
        if intent == "true_false":
            if not re.search(r"\b(phải không|có đúng không|đúng không|được không)\b", text, flags=re.IGNORECASE):
                text += " phải không?"
        elif intent == "advice":
            text += "?"
        elif intent == "law_reference":
            # chỉ thêm nếu chưa có cụm "theo quy định"
            if not re.search(r"\btheo quy định\b", text, flags=re.IGNORECASE):
                text += " theo quy định pháp luật?"
            else:
                text += "?"
        elif intent == "punishment":
            text += " bị xử lý thế nào?"
        else:
            text += "?"

    # --- 5. Làm sạch dấu câu ---
    text = re.sub(r"[!?]{2,}", "?", text)
    text = text.replace(",,", ",").replace("..", ".")
    text = text.strip()

    return {
        "normalized_query": text,
        "intent_hint": intent,
        "original_query": original
    }