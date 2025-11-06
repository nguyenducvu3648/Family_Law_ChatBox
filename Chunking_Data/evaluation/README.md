# Evaluation Module - AI Review

Module đánh giá chất lượng chunking với AI (Gemini).

---

## Overview

Module `evaluation/` cung cấp công cụ tự động kiểm tra chất lượng chunks bằng Gemini AI, giúp phát hiện:

- Sai thứ tự (Chương/Điều/Khoản/Điểm nhảy cóc)
- Nhận diện nhầm (Khoản không đúng format `1.`, Điểm không đúng `a)`)
- Thiếu/bỏ sót nội dung
- Metadata không khớp
- Lỗi xử lý chuỗi điểm

---

## Files

### `ai_reviewer.py`

Core AI Review logic.

**Functions:**

```python
# Build payload với sampling thông minh
def build_review_payload(
    chunks: List[Dict],
    summary: Dict,
    raw_texts: List[str],
    sample_excerpts_chars: int = 2000,
    max_chunks_sample: int = 50
) -> Dict

# Gọi Gemini API để review
def call_gemini_review(
    payload: Dict,
    api_key: Optional[str] = None
) -> Dict

# Save issues report
def save_issues_report(review: Dict, output_path: str) -> None

# Print review summary
def print_review_summary(review: Dict, verbose: bool = False) -> None
```

**Constants:**

```python
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
GEMINI_PROMPT = """..."""  # Chi tiết prompt cho AI review
```

---

## Usage

### Command Line

```bash
# Basic AI review
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI

# Strict mode (chỉ lưu nếu AI OK)
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI \
    --strict-ok-only

# Customize sampling
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI \
    --max-chunks-sample 100 \
    --max-files-sample 3 \
    --sample-excerpts 3000
```

### Programmatic

```python
from Chunking_Data.evaluation.ai_reviewer import (
    build_review_payload,
    call_gemini_review,
    print_review_summary
)

# Prepare data
chunks = [...]  # Your chunks
summary = {"articles": 100, "clauses": 200, ...}
raw_texts = ["raw text 1", "raw text 2"]

# Build payload
payload = build_review_payload(chunks, summary, raw_texts)

# Call AI review
review = call_gemini_review(payload, api_key="your-key")

# Print results
print_review_summary(review, verbose=True)

# Check status
if review['status'] == 'ok':
    print("✅ Chunks passed AI review!")
else:
    print(f"⚠️ Found {len(review['issues'])} issues")
```

---

## Review Output Format

```json
{
  "status": "ok | issues_found",
  "confidence": 0.85,
  "issues": [
    {
      "id": "LHNVDG-D24-K2-a",
      "citation": "Điều 24 khoản 2 điểm a",
      "severity": "high | medium | low",
      "category": "ordering | regex | metadata | omission | points_chain | format | other",
      "message": "Mô tả vấn đề",
      "suggestion": "Cách khắc phục"
    }
  ],
  "notes": "Ghi chú bổ sung"
}
```

---

## Sampling Strategy

AI Review sử dụng sampling thông minh để tránh vượt token limit:

### 1. **Excerpt Sampling**

- Lấy max 3 files
- Mỗi file: đầu, giữa, cuối (chia đều chars)
- Total: ~2000 chars (configurable)

### 2. **Chunk Sampling**

- Mix: Article/Clause/Point chunks (chia đều)
- Max: 50 chunks (configurable)
- Shuffle để tránh bias

### 3. **Content Preview**

- Mỗi chunk: max 500 chars preview
- Format: `{id, metadata, content_preview}`

---

## Configuration

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your-gemini-api-key-here
```

### CLI Arguments

| Argument              | Type | Default | Description                       |
| --------------------- | ---- | ------- | --------------------------------- |
| `--AI`                | flag | False   | Bật AI review                     |
| `--api-key`           | str  | None    | Gemini API key (override env var) |
| `--sample-excerpts`   | int  | 2000    | Total chars for excerpts          |
| `--max-chunks-sample` | int  | 50      | Max chunks to sample              |
| `--max-files-sample`  | int  | 2       | Max files for raw text            |
| `--strict-ok-only`    | flag | False   | Chỉ lưu nếu AI confirm OK         |

---

## Issue Categories

| Category       | Description                       |
| -------------- | --------------------------------- |
| `ordering`     | Sai thứ tự Chương/Điều/Khoản/Điểm |
| `regex`        | Nhận diện nhầm pattern            |
| `metadata`     | Metadata không khớp với content   |
| `omission`     | Thiếu/bỏ sót nội dung             |
| `points_chain` | Lỗi xử lý chuỗi điểm              |
| `format`       | Lỗi format chunk                  |
| `other`        | Lỗi khác                          |

---

## Best Practices

### 1. **API Key Management**

```bash
# Luôn dùng .env file
echo "GEMINI_API_KEY=your-key" > .env

# Không commit API key vào git
echo ".env" >> .gitignore
```

### 2. **Sampling Size**

```bash
# Small dataset (< 100 chunks)
--max-chunks-sample 50 --sample-excerpts 2000

# Medium dataset (100-500 chunks)
--max-chunks-sample 100 --sample-excerpts 3000

# Large dataset (> 500 chunks)
--max-chunks-sample 150 --sample-excerpts 4000
```

### 3. **Error Handling**

```python
try:
    review = call_gemini_review(payload)
except RuntimeError as e:
    print(f"AI Review failed: {e}")
    # Fallback: manual review hoặc skip
```

---

## Troubleshooting

| Error                    | Solution                                          |
| ------------------------ | ------------------------------------------------- |
| `Missing GEMINI_API_KEY` | Set env var hoặc dùng `--api-key`                 |
| `JSON parse error`       | Giảm `--sample-excerpts` và `--max-chunks-sample` |
| `Token limit exceeded`   | Giảm sampling parameters                          |
| `API rate limit`         | Chờ hoặc upgrade API plan                         |

---

## Dependencies

```bash
pip install google-generativeai python-dotenv
```

---

## Quick Start Guide

### Setup

```bash
# 1. Install dependencies
pip install google-generativeai python-dotenv

# 2. Create .env file
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env

# 3. Run with AI Review
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI \
    --verbose
```

### Use Cases

**Development (Fast):**

```bash
# No AI review - fast iteration
python -m Chunking_Data.scripts.chunk_documents --category BDS
```

**Testing (Debug):**

```bash
# With AI review and verbose output
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI \
    --verbose
```

**Production (Quality Assured):**

```bash
# Strict mode - only save if AI confirms OK
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --AI \
    --strict-ok-only
```

---
