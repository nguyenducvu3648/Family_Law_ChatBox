#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Review Module - Gemini-powered Chunking Quality Assurance

This module provides intelligent review of chunking results using Google's Gemini AI.
It samples chunks and raw text excerpts to detect anomalies and quality issues.

Main Functions:
    - build_review_payload(): Prepare sampled data for AI review
    - call_gemini_review(): Call Gemini API to evaluate chunks

Usage:
    from Chunking_Data.evaluation.ai_reviewer import build_review_payload, call_gemini_review
    
    payload = build_review_payload(chunks, summary, raw_texts)
    review = call_gemini_review(payload, api_key="your-key")
    
    if review['status'] == 'ok':
        print("Chunks passed AI review!")
    else:
        print(f"Found {len(review['issues'])} issues")
"""

import os
import json
import random
from typing import List, Dict, Any, Optional

# Gemini model name (fixed as per requirements)
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

# Gemini prompt for chunking review
GEMINI_PROMPT = """Bạn là chuyên gia pháp điển hoá & kiểm thử dữ liệu luật.
Tôi gửi cho bạn:
1) Summary thống kê & cảnh báo từ bộ chunking.
2) Một số "excerpts" (trích đoạn nguyên văn) đại diện.
3) Danh mục chunks (id + metadata + content rút gọn).

Nhiệm vụ:
- PHÁT HIỆN BẤT THƯỜNG (anomalies) trong chunking, ví dụ:
  * Sai thứ tự/chưa "strict" (nhảy cóc Chương/Điều/Khoản/Điểm).
  * Nhận diện nhầm "Khoản" (không phải dạng `1.`) hoặc "Điểm" (không phải `a)`).
  * Không tiêm intro khoản vào điểm khi đã có chuỗi điểm.
  * Thiếu/bỏ sót nội dung so với excerpts.
  * Metadata không khớp: article_no/clause_no/point_letter/exact_citation.
  * Đóng mở chuỗi điểm sai (bắt đầu không phải "a)", chèn nội dung thường vào giữa).
  * Nội dung "Điều" không có khoản nhưng không sinh chunk intro.
- GỢI Ý SỬA: chỉ rõ vị trí (id / exact_citation), mô tả vấn đề, cách khắc phục.
- Nếu KHÔNG thấy vấn đề, xác nhận "ok" và nêu ngắn gọn cơ sở kết luận.

Hãy TRẢ LỜI **CHỈ** ở dạng JSON theo schema:
{
  "status": "ok" | "issues_found",
  "confidence": 0.0-1.0,
  "issues": [
    {
      "id": "chuỗi id chunk hoặc mô tả vị trí",
      "citation": "Điều ... khoản ... điểm ...",
      "severity": "low|medium|high",
      "category": "ordering|regex|metadata|omission|points_chain|format|other",
      "message": "Mô tả ngắn gọn vấn đề",
      "suggestion": "Cách sửa ngắn gọn"
    }
  ],
  "notes": "Ghi chú ngắn (tuỳ chọn)"
}
Trả JSON hợp lệ. Không giải thích ngoài JSON.
"""


def _shorten_text(s: str, max_len: int = 500) -> str:
    """Shorten text for display in payload
    
    Args:
        s: Text to shorten
        max_len: Maximum length (default: 500)
        
    Returns:
        Shortened text with ellipsis in middle if needed
    """
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2].rstrip()
    tail = s[- max_len // 2 :].lstrip()
    return head + "\n...\n" + tail


def build_review_payload(
    chunks: List[Dict[str, Any]], 
    summary: Dict[str, Any], 
    raw_texts: List[str],
    sample_excerpts_chars: int = 2000,
    max_chunks_sample: int = 50
) -> Dict[str, Any]:
    """Build payload for AI review with intelligent sampling to avoid token limits.
    
    This function creates a optimized payload for Gemini review by:
    1. Sampling excerpts from raw texts (distributed evenly)
    2. Sampling chunks intelligently (mix of article/clause/point types)
    3. Limiting total payload size to avoid API limits
    
    Args:
        chunks: List of all chunks
        summary: Overall statistics dict
        raw_texts: List of raw texts from sample files
        sample_excerpts_chars: Total characters for excerpts (divided among files)
        max_chunks_sample: Maximum number of chunks to sample
        
    Returns:
        Payload dict ready for Gemini API
        
    Example:
        >>> payload = build_review_payload(chunks, summary, raw_texts)
        >>> payload.keys()
        dict_keys(['summary', 'excerpts', 'chunks_preview', 'note'])
    """
    
    # 1. Sample excerpts from raw texts (divide chars evenly per file)
    excerpts_per_file = sample_excerpts_chars // max(len(raw_texts), 1)
    excerpts_list = []
    
    for i, raw_text in enumerate(raw_texts[:3]):  # Max 3 files
        if len(raw_text) <= excerpts_per_file:
            excerpts_list.append(f"File {i+1}:\n{raw_text}")
        else:
            # Sample 3 parts: beginning, middle, end
            k = excerpts_per_file // 3
            n = len(raw_text)
            excerpt = raw_text[:k] + "\n...\n" + raw_text[n//2 - k//2 : n//2 + k//2] + "\n...\n" + raw_text[-k:]
            excerpts_list.append(f"File {i+1}:\n{excerpt}")
    
    combined_excerpts = "\n\n".join(excerpts_list)
    
    # 2. Intelligent chunk sampling (prioritize potential issues)
    def lite(c):
        """Create lightweight chunk representation"""
        return {
            "id": c.get("id"),
            "metadata": c.get("metadata"),
            "content_preview": _shorten_text(c.get("content", ""), 500)
        }
    
    # Sampling strategy: mix of different chunk types
    sampled_chunks = []
    article_chunks = []
    clause_chunks = []
    point_chunks = []
    
    for c in chunks:
        metadata = c.get('metadata', {})
        if metadata.get('point_letter'):
            point_chunks.append(c)
        elif metadata.get('clause_no'):
            clause_chunks.append(c)
        elif metadata.get('article_no'):
            article_chunks.append(c)
    
    # Sample evenly from each type (max max_chunks_sample total)
    samples_per_type = max_chunks_sample // 3
    
    sampled_chunks.extend(random.sample(article_chunks, min(samples_per_type, len(article_chunks))))
    sampled_chunks.extend(random.sample(clause_chunks, min(samples_per_type, len(clause_chunks))))
    sampled_chunks.extend(random.sample(point_chunks, min(samples_per_type, len(point_chunks))))
    
    # Fill remaining slots if needed
    remaining = max_chunks_sample - len(sampled_chunks)
    if remaining > 0:
        other_chunks = [c for c in chunks if c not in sampled_chunks]
        sampled_chunks.extend(random.sample(other_chunks, min(remaining, len(other_chunks))))
    
    # Shuffle to avoid bias
    random.shuffle(sampled_chunks)
    chunks_lite = [lite(c) for c in sampled_chunks]
    
    return {
        "summary": summary,
        "excerpts": combined_excerpts,
        "chunks_preview": chunks_lite,
        "note": f"Sampled {len(chunks_lite)}/{len(chunks)} chunks from {len(raw_texts)} files for AI review"
    }


def call_gemini_review(payload: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call Gemini AI to review chunking quality.
    
    This function sends the payload to Gemini and receives a structured JSON response
    with status, confidence, and any detected issues.
    
    Args:
        payload: Review payload from build_review_payload()
        api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
        
    Returns:
        Review dict with keys:
            - status: 'ok' or 'issues_found'
            - confidence: float 0.0-1.0
            - issues: list of issue dicts
            - notes: optional notes string
            
    Raises:
        RuntimeError: If API key is missing or API call fails
        
    Example:
        >>> review = call_gemini_review(payload, api_key="your-key")
        >>> if review['status'] == 'ok':
        ...     print("Chunks are valid!")
        >>> else:
        ...     print(f"Found {len(review['issues'])} issues")
    """
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. "
            "Please set GEMINI_API_KEY environment variable or pass api_key parameter."
        )
    
    # Import Gemini library (only when needed)
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. "
            "Install with: pip install google-generativeai"
        )
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config)
    
    # Prepare prompt
    prompt_parts = [
        {"role": "user", "parts": [{"text": GEMINI_PROMPT}]},
        {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
    ]
    
    # Call Gemini
    try:
        resp = model.generate_content(prompt_parts)
        raw = getattr(resp, "text", None) or (
            resp.candidates and resp.candidates[0].content.parts[0].text
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")
    
    if not raw:
        raise RuntimeError("Gemini returned empty response")
    
    # Parse JSON response
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON is not an object")
        
        # Ensure required fields
        data.setdefault("status", "issues_found")
        data.setdefault("issues", [])
        data.setdefault("confidence", 0.0)
        
        return data
        
    except Exception as e:
        # If JSON parsing fails, return error as issue
        return {
            "status": "issues_found",
            "confidence": 0.0,
            "issues": [{
                "id": "PARSER",
                "citation": "",
                "severity": "high",
                "category": "other",
                "message": f"Cannot parse JSON from Gemini: {e}",
                "suggestion": "Re-run with different sampling or check API response."
            }],
            "notes": raw[:2000]  # Include first 2000 chars of raw response
        }


def save_issues_report(review: Dict[str, Any], output_path: str) -> None:
    """Save AI review issues to JSON file.
    
    Args:
        review: Review dict from call_gemini_review()
        output_path: Path to save issues JSON file
        
    Example:
        >>> save_issues_report(review, "data/chunks.issues.json")
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(review, f, ensure_ascii=False, indent=2)


def print_review_summary(review: Dict[str, Any], verbose: bool = False) -> None:
    """Print formatted review summary to console.
    
    Args:
        review: Review dict from call_gemini_review()
        verbose: If True, print detailed issues
        
    Example:
        >>> print_review_summary(review, verbose=True)
    """
    status = review.get("status", "unknown")
    confidence = review.get("confidence", 0.0)
    issues = review.get("issues", [])
    notes = review.get("notes", "")
    
    print(f"\n{'='*80}")
    print(f"AI REVIEW RESULTS")
    print(f"{'='*80}")
    print(f"Status: {status.upper()}")
    print(f"Confidence: {confidence:.2%}")
    
    if notes:
        print(f"Notes: {notes[:200]}...")
    
    if issues:
        print(f"\nFound {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            severity = issue.get('severity', '?')
            category = issue.get('category', 'other')
            citation = issue.get('citation') or issue.get('id') or 'N/A'
            message = issue.get('message', 'No message')
            
            print(f"\n{i}. [{severity.upper()}] ({category}) {citation}")
            print(f"   {message}")
            
            if verbose and issue.get('suggestion'):
                print(f"   → Suggestion: {issue['suggestion']}")
    else:
        print("\n✅ No issues found - chunks passed AI review!")
    
    print(f"{'='*80}\n")

