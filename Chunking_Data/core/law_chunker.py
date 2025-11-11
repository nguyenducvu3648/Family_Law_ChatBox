#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Law Chunker Module
==================

Module chia văn bản luật Việt Nam thành chunks theo cấu trúc pháp điển:
Chương > Mục > Điều > Khoản > Điểm

Features:
  - 2-pass parsing (pre-scan + strict parsing)
  - Clause intro injection vào points
  - Article intro handling
  - Citation generation
  - Comprehensive statistics
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple


# ==================== ROMAN NUMERAL UTILITIES ====================

ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}


def roman_to_int(s: str) -> Optional[int]:
    """Chuyển đổi số La Mã sang số nguyên"""
    s = s.upper().strip()
    if not s or any(ch not in ROMAN_MAP for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def chapter_to_int(s: str) -> Optional[int]:
    """Chuyển đổi số Chương (La Mã hoặc Ả Rập) sang số nguyên"""
    s = s.strip().upper()
    if not s:
        return None
    
    # Thử số La Mã trước
    roman_num = roman_to_int(s)
    if roman_num is not None:
        return roman_num
    
    # Thử số Ả Rập
    try:
        return int(s)
    except ValueError:
        return None


# ==================== TEXT NORMALIZATION ====================

def normalize_lines(text: str) -> List[str]:
    """
    Chuẩn hóa lines để parsing chính xác hơn.
    
    - NFC unicode normalization (fix dấu tiếng Việt)
    - Replace no-break spaces với spaces thường
    - Strip trailing whitespace
    - Remove BOM
    
    Args:
        text: Raw text từ docx
    
    Returns:
        List các dòng đã chuẩn hóa
    """
    if text is None:
        return []
    
    # Unicode normalization (composed form)
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()
    
    out: List[str] = []
    for ln in lines:
        if ln is None:
            continue
        # Replace các loại no-break space
        ln = ln.replace('\u00A0', ' ').replace('\u202F', ' ').replace('\u2009', ' ')
        # Remove BOM
        ln = ln.replace('\ufeff', '')
        # Trim trailing whitespace
        ln = re.sub(r'\s+$', '', ln)
        out.append(ln)
    
    return out


# ==================== INTRO TEXT DETECTION ====================

INTRO_CUE_PAT = re.compile(
    r'(sau đây|bao gồm|gồm các|quy định như sau)\s*:\s*$',
    re.IGNORECASE | re.UNICODE
)


def is_intro_text_for_clauses(text: str) -> bool:
    """
    Heuristic: Kiểm tra xem intro Điều có áp dụng cho các khoản không.
    
    Intro Điều dành cho khoản nếu:
      - Kết thúc bằng ':' HOẶC
      - Chứa cụm: 'sau đây:', 'bao gồm:', 'quy định như sau:', ...
    
    Args:
        text: Nội dung intro cần kiểm tra
    
    Returns:
        True nếu intro dành cho khoản, False nếu không
    """
    if not text:
        return False
    t = text.strip()
    if t.endswith(':'):
        return True
    if INTRO_CUE_PAT.search(t):
        return True
    return False


# ==================== MAIN CHUNKING FUNCTION ====================

def chunk_law_document(
    text: str,
    law_id: str = "LAW",
    law_no: str = "",
    law_title: str = "",
    issued_date: str = "",
    effective_date: str = "",
    expiry_date: Optional[str] = None,
    signer: str = "",
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Chia văn bản luật thành chunks theo cấu trúc pháp điển.
    
    Workflow:
      1. Pass 1 (Pre-scan): Scan toàn bộ để xác định chapters_set, articles_set
      2. Pass 2 (Strict parsing): Parse từng dòng theo state machine
      3. Flush chunks: Đóng khoản/điều/điểm theo đúng thứ tự
    
    Args:
        text: Raw text từ docx
        law_id: ID của văn bản luật (VD: "LHNVDG")
        law_no: Số hiệu luật (VD: "52/2014/QH13")
        law_title: Tên luật
        issued_date: Ngày ban hành (YYYY-MM-DD)
        effective_date: Ngày có hiệu lực (YYYY-MM-DD)
        expiry_date: Ngày hết hiệu lực (optional)
        signer: Người ký
        verbose: In log chi tiết
    
    Returns:
        Tuple[chunks, stats] where:
          - chunks: List[Dict] - Danh sách chunks
          - stats: Dict - Statistics (articles, clauses, points, citations, etc.)
    """
    if verbose:
        print("   🔄 Chunking law document with strict parsing...")
    
    lines = normalize_lines(text)
    
    # ===== REGEX PATTERNS (đầu dòng) =====
    ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]?\s*(.*)$', re.UNICODE)
    CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+|\d+)\s*:?\s*(.*)$', re.UNICODE | re.IGNORECASE)
    SECTION_RE = re.compile(r'^Mục\s+(\d+)[:\s]*(.*)$', re.UNICODE | re.IGNORECASE)
    CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)
    POINT_RE = re.compile(r'^\s*([a-zA-ZđĐ])[\)\.]\s+(.*)$', re.UNICODE)
    
    # ===== PASS 1: PRE-SCAN CHƯƠNG/ĐIỀU =====
    chapters_nums, articles_nums = [], []
    chapters_labels, article_titles_seen = [], []
    expecting_chapter_title = False
    roman_current = None
    
    for raw in lines:
        line = raw
        if not line:
            continue
        
        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or SECTION_RE.match(line) or 
                    CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapters_labels[-1] = lbl
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False
        
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            n = chapter_to_int(m_ch.group(1))
            if n:
                chapters_nums.append(n)
                title = (m_ch.group(2) or "").strip()
                lbl = f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else "")
                chapters_labels.append(lbl)
                if not title:
                    expecting_chapter_title = True
                    roman_current = m_ch.group(1).strip()
                else:
                    expecting_chapter_title = False
            continue
        
        m_art = ARTICLE_RE.match(line)
        if m_art:
            num = int(m_art.group(1))
            articles_nums.append(num)
            article_titles_seen.append((m_art.group(2) or "").strip())
            continue
    
    chapters_set, articles_set = set(chapters_nums), set(articles_nums)
    
    # ===== PASS 2: STRICT PARSING =====
    chunks = []
    citations = []
    stats = {"articles": 0, "article_intro": 0, "clauses": 0, "points": 0}
    chapters_seen_labels = []
    
    # State variables
    chapter_label: Optional[str] = None
    expecting_chapter_title: bool = False
    roman_current: Optional[str] = None
    chapter_number: Optional[int] = None
    section_label: Optional[str] = None
    
    article_no: Optional[int] = None
    article_title: str = ""
    expecting_article_title: bool = False
    
    article_intro_buf: str = ""
    article_has_any_chunk: bool = False
    
    clause_no: Optional[int] = None
    clause_buf: str = ""
    clause_intro_current: Optional[str] = None
    article_clause_intro_current: Optional[str] = None
    
    in_points: bool = False
    point_letter: Optional[str] = None
    point_buf: str = ""
    
    expected_chapter: Optional[int] = None
    expected_article: Optional[int] = None
    seeking_article: bool = False
    
    # ===== HELPER FUNCTIONS =====
    
    def build_article_header(a_no: int, a_title: str) -> str:
        """Tạo header Điều"""
        t = (a_title or "").strip()
        return f"Điều {a_no}" + (f". {t}" if t else "")
    
    def flush_article_intro():
        """Flush chunk intro Điều (khi Điều không có khoản)"""
        nonlocal article_intro_buf, article_has_any_chunk
        content = article_intro_buf.strip()
        if not content:
            return
        
        cid = f"{law_id}-D{article_no}"
        exact = f"Điều {article_no}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "issued_date": issued_date,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "signer": signer,
            "chapter": chapter_label,
            "chapter_number": chapter_number,
            "chapter_title": chapter_label,
            "section": section_label or None,
            "article_no": article_no,
            "article_title": article_title or None,
            "clause_no": None,
            "clause_intro": None,
<<<<<<< HEAD
            "point_id": None,
=======
            "point_id": f"dieu_{article_no}",
>>>>>>> 985580bf68add13b2bc7f26f77585e9417bff953
            "point_letter": None,
            "exact_citation": exact
        }
        title_line = f"Điều {article_no}. {article_title}".strip() if article_title else f"Điều {article_no}"
        chunks.append({
            "id": cid,
            "content": f"{title_line}\n{content}",
            "metadata": meta
        })
        stats["article_intro"] += 1
        citations.append(exact)
        article_intro_buf = ""
        article_has_any_chunk = True
    
    def flush_clause():
        """Flush chunk khoản"""
        nonlocal clause_buf
        content = (clause_buf or "").strip()
        if not content:
            return
        
        cid = f"{law_id}-D{article_no}-K{clause_no}"
        exact = f"Điều {article_no} khoản {clause_no}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "issued_date": issued_date,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "signer": signer,
            "chapter": chapter_label,
            "chapter_number": chapter_number,
            "chapter_title": chapter_label,
            "section": section_label or None,
            "article_no": article_no,
            "article_title": article_title or None,
            "clause_no": clause_no,
            "clause_intro": clause_intro_current or None,
<<<<<<< HEAD
            "point_id": None,
=======
            "point_id": f"dieu_{article_no}_khoan_{clause_no}",
>>>>>>> 985580bf68add13b2bc7f26f77585e9417bff953
            "point_letter": None,
            "exact_citation": exact
        }
        art_hdr = build_article_header(article_no, article_title)
        full_content = f"{art_hdr} Khoản {clause_no}. {content}"
        
        chunks.append({"id": cid, "content": full_content, "metadata": meta})
        stats["clauses"] += 1
        citations.append(exact)
    
    def flush_point():
        """Flush chunk điểm"""
        nonlocal point_buf
        content = (point_buf or "").strip()
        if not content:
            return
        
        letter = point_letter.lower()
        cid = f"{law_id}-D{article_no}-K{clause_no}-{letter}"
        exact = f"Điều {article_no} khoản {clause_no} điểm {letter}."
        point_id = f"dieu_{article_no}_khoan_{clause_no}_diem_{letter}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "issued_date": issued_date,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "signer": signer,
            "chapter": chapter_label,
            "chapter_number": chapter_number,
            "chapter_title": chapter_label,
            "section": section_label or None,
            "article_no": article_no,
            "article_title": article_title or None,
            "clause_no": clause_no,
            "clause_intro": clause_intro_current or None,
<<<<<<< HEAD
            "point_id": point_id,
            "point_letter": letter,
=======
    "point_id": f"dieu_{article_no}_khoan_{clause_no}_diem_{letter}",  # Luôn có cho tất cả chunk types
    "point_letter": letter,
>>>>>>> 985580bf68add13b2bc7f26f77585e9417bff953
            "exact_citation": exact
        }
        art_hdr = build_article_header(article_no, article_title)
        
        # Tiêm clause intro vào điểm
        if clause_intro_current:
            intro = clause_intro_current.rstrip().rstrip(':')
            full_content = f"{art_hdr} Khoản {clause_no}. {intro}, điểm {letter}.\n{content}"
        else:
            full_content = f"{art_hdr} Khoản {clause_no}, điểm {letter}. {content}"
        
        chunks.append({"id": cid, "content": full_content, "metadata": meta})
        stats["points"] += 1
        citations.append(exact)
    
    def close_clause():
        """Đóng khoản hiện tại"""
        nonlocal clause_no, clause_buf, in_points, point_letter, point_buf
        nonlocal article_has_any_chunk, clause_intro_current
        
        if clause_no is None:
            return
        
        if in_points and point_letter:
            flush_point()
        elif clause_buf.strip():
            flush_clause()
        
        article_has_any_chunk = True
        clause_no, clause_buf, in_points = None, "", False
        point_letter, point_buf = None, ""
        clause_intro_current = None
    
    def close_article_if_needed():
        """Đóng Điều nếu cần (flush intro nếu chưa có chunk nào)"""
        nonlocal article_intro_buf, article_has_any_chunk
        if not article_has_any_chunk and article_intro_buf.strip():
            flush_article_intro()
        article_intro_buf = ""
        article_has_any_chunk = False
    
    # ===== MAIN PARSING LOOP =====
    
    if verbose:
        print(f"   📊 Processing {len(lines):,} lines...")
    
    for line in lines:
        if not line:
            continue
        
        # Seeking article mode (skip lines until find expected article)
        if seeking_article:
            m_art_seek = ARTICLE_RE.match(line)
            if m_art_seek:
                a_no = int(m_art_seek.group(1))
                if a_no == expected_article:
                    seeking_article = False
                    close_clause()
                    if article_no is not None:
                        close_article_if_needed()
                    article_no = a_no
                    article_title = (m_art_seek.group(2) or "").strip()
                    stats["articles"] += 1
                    if not article_title:
                        expecting_article_title = True
                    expected_article = a_no + 1
                    clause_no, clause_buf = None, ""
                    in_points, point_letter, point_buf = False, None, ""
                    clause_intro_current = None
                    article_clause_intro_current = None
                    continue
            continue
        
        # Expecting article title
        if expecting_article_title:
            if not any(regex.match(line) for regex in [CHAPTER_RE, SECTION_RE, CLAUSE_RE, POINT_RE, ARTICLE_RE]):
                article_title = line
                expecting_article_title = False
                continue
            else:
                expecting_article_title = False
        
        # Expecting chapter title
        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or SECTION_RE.match(line) or 
                    CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapter_label = lbl
                if lbl not in chapters_seen_labels:
                    chapters_seen_labels.append(lbl)
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False
        
        # MỤC - Track section label
        m_sec = SECTION_RE.match(line)
        if m_sec:
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False
            
            sec_no = m_sec.group(1).strip()
            sec_title = (m_sec.group(2) or "").strip()
            section_label = f"Mục {sec_no}" + (f" – {sec_title}" if sec_title else "")
            continue
        
        # CHƯƠNG
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False
            article_clause_intro_current = None
            article_has_any_chunk = False
            section_label = None
            
            chapter_str = m_ch.group(1).strip()
            ch_num = chapter_to_int(chapter_str) or 0
            ch_title = (m_ch.group(2) or "").strip()
            lbl = f"Chương {chapter_str}" + (f" – {ch_title}" if ch_title else "")
            
            if not ch_title:
                expecting_chapter_title = True
                roman_current = chapter_str
                chapter_number = ch_num
            else:
                chapter_label = lbl
                chapter_number = ch_num
                if lbl not in chapters_seen_labels:
                    chapters_seen_labels.append(lbl)
                expecting_chapter_title = False
            
            if expected_chapter is None:
                expected_chapter = ch_num + 1
            else:
                if ch_num == expected_chapter:
                    expected_chapter = ch_num + 1
                elif ch_num > expected_chapter:
                    if expected_chapter not in chapters_set:
                        break
                    continue
                else:
                    continue
            
            continue
        
        # ĐIỀU
        m_art = ARTICLE_RE.match(line)
        if m_art:
            a_no = int(m_art.group(1))
            a_title = (m_art.group(2) or "").strip()
            
            if expected_article is None:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None:
                    close_article_if_needed()
                article_no = a_no
                article_title = a_title
                stats["articles"] += 1
                if not article_title:
                    expecting_article_title = True
                clause_no, clause_buf = None, ""
                in_points, point_letter, point_buf = False, None, ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            elif a_no == expected_article:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None:
                    close_article_if_needed()
                article_no = a_no
                article_title = a_title
                stats["articles"] += 1
                if not article_title:
                    expecting_article_title = True
                clause_no, clause_buf = None, ""
                in_points, point_letter, point_buf = False, None, ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            elif a_no > expected_article:
                if expected_article not in articles_set:
                    break
                else:
                    seeking_article = True
                    continue
            else:
                continue
        
        if article_no is None:
            continue
        
        # KHOẢN
        m_k = CLAUSE_RE.match(line)
        if m_k and m_k.group(1).isdigit():
            # Xử lý intro Điều
            if article_intro_buf.strip() and is_intro_text_for_clauses(article_intro_buf):
                article_clause_intro_current = article_intro_buf.strip()
                article_intro_buf = ""
            elif article_intro_buf.strip():
                flush_article_intro()
                article_has_any_chunk = True
            
            close_clause()
            clause_no = int(m_k.group(1))
            clause_buf = (m_k.group(2) or "").strip()
            in_points, point_letter, point_buf = False, None, ""
            clause_intro_current = None
            continue
        
        # ĐIỂM
        m_p = POINT_RE.match(line)
        if m_p:
            # Nếu chưa có khoản, tạo khoản 1 ẩn
            if clause_no is None:
                clause_no = 1
                clause_buf = ""
                clause_intro_current = None
            
            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()
            
            if not in_points:
                if letter != 'a':
                    # Không phải điểm, nối vào clause
                    clause_buf += ("\n" if clause_buf else "") + f"{letter}. {text}"
                    continue
                # Bắt đầu chuỗi điểm với 'a)'
                clause_intro_current = clause_buf.strip() if clause_buf.strip() else None
                clause_buf = ""
                in_points = True
                point_letter = letter
                point_buf = text
                continue
            
            # Đang trong chuỗi điểm
            if point_letter:
                flush_point()
            in_points = True
            point_letter = letter
            point_buf = text
            continue
        
        # Nội dung kéo dài
        if clause_no is not None:
            if in_points and point_letter:
                point_buf += ("\n" if point_buf else "") + line
            else:
                clause_buf += ("\n" if clause_buf else "") + line
        else:
            article_intro_buf += ("\n" if article_intro_buf else "") + line
    
    # Kết thúc parsing
    close_clause()
    if article_no is not None:
        close_article_if_needed()
    
    # Filter chunks (loại bỏ chunks quá ngắn)
    valid_chunks = []
    for chunk in chunks:
        content = chunk['content'].strip()
        if len(content) > 50:
            valid_chunks.append(chunk)
    
    if verbose:
        print(f"   ✅ Created {len(valid_chunks)} chunks")
        print(f"   📊 Stats - Articles: {stats['articles']}, "
              f"Article Intro: {stats['article_intro']}, "
              f"Clauses: {stats['clauses']}, "
              f"Points: {stats['points']}")
        print(f"   📂 Chapters seen: {len(chapters_seen_labels)}, "
              f"Citations: {len(citations)}")
    
    # Return với stats đầy đủ
    return valid_chunks, {
        "chapters_seen": chapters_seen_labels,
        "articles": stats["articles"],
        "article_intro": stats["article_intro"],
        "clauses": stats["clauses"],
        "points": stats["points"],
        "citations": citations,
        "total_chunks": len(valid_chunks)
    }

