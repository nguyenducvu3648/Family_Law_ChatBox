#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Reader Module
======================

Module đọc file .doc/.docx với nhiều phương pháp fallback.

Supported methods:
  1. python-docx (primary for .docx)
  2. docx2txt (fallback for .docx)
  3. textract (for both .doc and .docx)
  4. pypandoc (universal converter)
  5. antiword (for .doc via subprocess)
"""

import os
from typing import Optional


def read_docx(file_path: str, verbose: bool = True) -> str:
    """
    Đọc file .doc hoặc .docx và trả về text content.
    
    Args:
        file_path: Đường dẫn đến file docx
        verbose: In log chi tiết quá trình đọc
    
    Returns:
        Text content của document
    
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        ValueError: Nếu không thể đọc được content từ file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Safe path cho console output
    safe_path = file_path.encode('ascii', 'ignore').decode('ascii') or file_path
    if verbose:
        print(f"   📖 Reading file: {safe_path}")
    
    # Xác định file extension
    file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
    
    # Method 1: python-docx (primary cho .docx)
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join((p.text or "").strip() for p in doc.paragraphs)
        if text and len(text.strip()) > 10:
            if verbose:
                print(f"   ✅ Read {len(text):,} chars using python-docx")
            return text
        elif verbose:
            print(f"   ⚠️  python-docx returned minimal content, trying alternatives...")
    except Exception as e:
        if verbose:
            safe_e = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
            print(f"   ⚠️  python-docx failed: {safe_e}")
    
    # Method 2: docx2txt (fallback cho .docx)
    try:
        import docx2txt
        text = docx2txt.process(file_path)
        if text and len(text.strip()) > 10:
            if verbose:
                print(f"   ✅ Read {len(text):,} chars using docx2txt")
            return text
        elif verbose:
            print(f"   ⚠️  docx2txt returned minimal content")
    except Exception as e:
        if verbose:
            safe_e = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
            print(f"   ⚠️  docx2txt failed: {safe_e}")
    
    # Method 3: textract (cho cả .doc và .docx)
    try:
        import textract
        text = textract.process(file_path).decode('utf-8', errors='ignore')
        if text and len(text.strip()) > 10:
            if verbose:
                print(f"   ✅ Read {len(text):,} chars using textract")
            return text
        elif verbose:
            print(f"   ⚠️  textract returned minimal content")
    except Exception as e:
        if verbose:
            safe_e = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
            print(f"   ⚠️  textract failed: {safe_e}")
    
    # Method 4: pypandoc (universal)
    try:
        import pypandoc
        text = pypandoc.convert_file(file_path, 'plain', extra_args=['--wrap=none'])
        if text and len(text.strip()) > 10:
            if verbose:
                print(f"   ✅ Read {len(text):,} chars using pypandoc")
            return text
        elif verbose:
            print(f"   ⚠️  pypandoc returned minimal content")
    except Exception as e:
        if verbose:
            safe_e = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
            print(f"   ⚠️  pypandoc failed: {safe_e}")
    
    # Method 5: antiword (chỉ cho .doc)
    if file_ext == 'doc':
        try:
            import subprocess
            result = subprocess.run(
                ['antiword', file_path, '-w', '0'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout and len(result.stdout.strip()) > 10:
                if verbose:
                    print(f"   ✅ Read {len(result.stdout):,} chars using antiword")
                return result.stdout
            elif verbose:
                print(f"   ⚠️  antiword failed or returned minimal content")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            if verbose:
                safe_e = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                print(f"   ⚠️  antiword failed: {safe_e}")
    
    # Nếu tất cả methods đều thất bại
    raise ValueError(f"All reading methods failed for file: {safe_path}")


def get_supported_extensions():
    """Trả về list các file extensions được hỗ trợ"""
    return ['.doc', '.docx']


def is_supported_file(file_path: str) -> bool:
    """Kiểm tra xem file có được hỗ trợ không"""
    ext = os.path.splitext(file_path.lower())[1]
    return ext in get_supported_extensions()

