#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Law ID Generator Module
========================

Module tự động sinh ID cho các văn bản luật dựa trên tên file.

Features:
  - Mapping thông minh các loại luật phổ biến
  - Xử lý luật sửa đổi/bổ sung (LSĐBS prefix)
  - Fallback tạo ID từ chữ cái đầu
  - Hỗ trợ cả tên có dấu và không dấu
"""

import re
import unicodedata
from typing import Optional


# Mapping các loại luật phổ biến → Law ID
LAW_MAPPINGS = {
    # Bất động sản
    'kinh doanh bất động sản': 'LKBDS',
    'nhà ở': 'LNHAO',
    'đất đai': 'LDATDAI',
    
    # Đầu tư
    'đầu tư': 'LDAUTU',
    'đầu tư công': 'LDAUTUCONG',
    'đầu tư theo phương thức đối tác công tư': 'LDAUTUPPPCT',
    
    # Thuế
    'thuế sử dụng đất nông nghiệp': 'LTSDDNONGNGHIEP',
    'thuế sử dụng đất phi nông nghiệp': 'LTSDDPHINONGNGHIEP',
    
    # Xây dựng
    'xây dựng': 'LXAYDUNG',
    
    # Hôn nhân & Gia đình
    'hôn nhân và gia đình': 'LHNVDG',
    'hôn nhân gia đình': 'LHNVDG',
    
    # Sở hữu trí tuệ
    'sở hữu trí tuệ': 'LSHTT',
    
    # Doanh nghiệp
    'doanh nghiệp': 'LDOANHNGHIEP',
    'công ty': 'LCONGTY',
    
    # Thương mại
    'thương mại': 'LTHUONGMAI',
    
    # Dân sự
    'dân sự': 'LDANSU',
    'quyền dân sự': 'LQUYENDANSU',
}

# Stop words để bỏ qua khi tạo ID từ chữ cái đầu
STOP_WORDS = {
    'số', 'và', 'theo', 'phương', 'thức', 'đối', 'tác', 'công', 'tư',
    'luật', 'văn', 'bản', 'hợp', 'nhất', 'năm', 'qđ', 'tt', 'bh', 
    'vbh', 'vbhn', 'vpqh', 'của', 'về', 'các', 'trong', 'cho'
}

# Keywords cho luật sửa đổi/bổ sung
AMENDMENT_KEYWORDS = ['sửa đổi', 'bổ sung', 'sửa đổi, bổ sung']


def normalize_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, replace _ với space"""
    return text.lower().replace('_', ' ').strip()


def remove_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt"""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')


def extract_important_words(text: str) -> list:
    """Trích xuất các từ quan trọng từ text"""
    words = re.split(r'[_\s]+', text)
    important_words = []
    
    for word in words:
        word_lower = word.lower()
        # Bỏ qua số, từ dừng, từ quá ngắn
        if len(word) <= 1 or word_lower in STOP_WORDS or word.isdigit():
            continue
        # Bỏ qua pattern số như 2020_QH14
        if '_' in word and any(part.isdigit() for part in word.split('_')):
            continue
        important_words.append(word)
    
    return important_words


def generate_law_id_from_name(name: str) -> str:
    """
    Tạo law_id từ tên đã chuẩn hóa.
    
    Args:
        name: Tên văn bản luật (đã loại bỏ extension)
    
    Returns:
        Law ID string (VD: "LKBDS", "LHNVDG")
    """
    name_normalized = normalize_text(name)
    
    # Thử match với LAW_MAPPINGS
    for key, value in LAW_MAPPINGS.items():
        # Chuẩn hóa key để matching
        key_normalized = remove_accents(key).lower()
        name_no_accent = remove_accents(name_normalized)
        
        # Check exact match
        if key in name_normalized or key_normalized in name_no_accent:
            return value
        
        # Check từng từ trong key (chỉ nếu key có ít nhất 2 từ)
        key_words = key.split()
        if len(key_words) >= 2:
            key_words_normalized = [remove_accents(w).lower() for w in key_words]
            # Check whole word với boundary
            cond1 = all(re.search(r'\b' + re.escape(word) + r'\b', name_normalized) 
                       for word in key_words)
            cond2 = all(re.search(r'\b' + re.escape(word) + r'\b', name_no_accent) 
                       for word in key_words_normalized)
            if cond1 or cond2:
                return value
    
    # Nếu không match được, tạo ID từ chữ cái đầu
    important_words = extract_important_words(name)
    
    if important_words:
        # Lấy chữ cái đầu của 2-4 từ quan trọng đầu tiên
        first_letters = ''.join(w[0].upper() for w in important_words[:4])
        result = f"L{first_letters}"
        return result[:8]  # Giới hạn độ dài tối đa 8 ký tự
    
    # Fallback cuối cùng
    words = re.split(r'[_\s]+', name)
    first_letters = ''.join(w[0].upper() for w in words[:3] 
                           if len(w) > 1 and not w.isdigit())
    return f"L{first_letters[:6]}"


def generate_law_id(file_name: str) -> str:
    """
    Tự động sinh law_id từ tên file.
    
    Xử lý:
      - Luật gốc: "Luật Sở hữu trí tuệ.docx" → "LSHTT"
      - Luật sửa đổi: "Luật sửa đổi Luật SHTT.docx" → "LSĐBSLSHTT"
    
    Args:
        file_name: Tên file (có thể có hoặc không có extension)
    
    Returns:
        Law ID string
    """
    # Loại bỏ extension
    name = file_name.replace('.docx', '').replace('.doc', '').strip()
    name_normalized = normalize_text(name)
    
    # Kiểm tra luật sửa đổi/bổ sung
    is_amendment = any(keyword in name_normalized for keyword in AMENDMENT_KEYWORDS)
    
    if is_amendment:
        # Tìm tên luật gốc (phần sau keyword)
        amendment_part = None
        for keyword in AMENDMENT_KEYWORDS:
            if keyword in name_normalized:
                parts = name_normalized.split(keyword, 1)
                if len(parts) > 1:
                    amendment_part = parts[1].strip()
                    break
        
        if amendment_part:
            # Tạo ID cho luật gốc
            base_law_id = generate_law_id_from_name(amendment_part)
            if base_law_id and base_law_id != 'LUNKNOWN':
                return f'LSĐBS{base_law_id}'
        
        # Fallback
        return 'LSĐBS'
    
    # Xử lý các trường hợp đặc biệt
    if 'luật số' in name_normalized and 'qh' in name_normalized:
        if 'xây dựng' in name_normalized:
            return 'LXAYDUNG'
    
    if 'văn bản hợp nhất' in name_normalized:
        if 'đầu tư' in name_normalized:
            return 'LDAUTU'
        if 'xây dựng' in name_normalized:
            return 'LXAYDUNG'
    
    if name_normalized.startswith('vbhn') or 'vbhn' in name_normalized:
        return 'LXAYDUNG'
    
    # Tạo ID thông thường
    return generate_law_id_from_name(name)


def validate_law_id(law_id: str) -> bool:
    """
    Kiểm tra tính hợp lệ của law_id.
    
    Args:
        law_id: Law ID cần kiểm tra
    
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not law_id or len(law_id) < 2:
        return False
    
    # Law ID phải bắt đầu bằng 'L'
    if not law_id.startswith('L'):
        return False
    
    # Chỉ chứa chữ cái và ký tự đặc biệt (Đ)
    if not re.match(r'^L[A-ZĐSB]+$', law_id):
        return False
    
    return True


# Alias cho backward compatibility
def generate_law_id_auto(file_name: str) -> str:
    """Alias của generate_law_id()"""
    return generate_law_id(file_name)

