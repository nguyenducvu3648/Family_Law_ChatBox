# Chatbot Luật Hôn Nhân & Gia Đình 2014

## Tổng quan

Đây là hệ thống chatbot chuyên về Luật Hôn Nhân & Gia Đình Việt Nam 2014, sử dụng kiến trúc RAG (Retrieval-Augmented Generation) kết hợp với nhiều phương pháp tìm kiếm và xếp hạng để cung cấp câu trả lời chính xác dựa trên các điều luật.

## Cấu trúc thư mục và chức năng

### 📁 Core Architecture

```
AI_Agent/Honnhan_Agent/
├── main.py              # Entry point chính, khởi tạo UI
├── core/                # Cấu hình và logging
│   ├── config.py        # Cấu hình môi trường, API keys, prompts
│   └── logging_setup.py # Thiết lập logging và monitoring
├── models/              # Mô hình AI và kết nối database
│   └── models.py        # Khởi tạo Qdrant, embedding, Gemini, rerank models
├── agents/              # Xử lý ý định và sinh câu trả lời
│   ├── intent.py        # Phân tích ý định người dùng
│   └── llm.py           # Streaming response từ Gemini
├── retrieval/           # Tìm kiếm và trích xuất tài liệu
│   ├── search.py        # Hybrid search (BM25 + embedding + rerank)
│   ├── fetch.py         # Tìm kiếm theo filter cụ thể
│   └── bm25_store.py    # Khởi tạo BM25 index toàn cục
├── tools/               # Utilities và reranking
│   └── tools.py         # BAAI reranking, tokenization, encoding
├── services/            # Xử lý prompt và render
│   ├── prompt.py        # Xây dựng prompt cho LLM
│   └── render.py        # Chuyển đổi docs thành markdown
├── memory/              # Cache và lưu trữ tạm
│   └── cache.py         # TTL cache cho embedding và search
├── utils/               # Utilities chung
│   └── utils.py         # Normalization, legal detection
└── api/                 # Giao diện người dùng
    └── ui.py            # Gradio interface và xử lý request
```

## Luồng hoạt động chi tiết

### 1. 🚀 Khởi tạo hệ thống (main.py)

```python
# Import tất cả modules theo tầng kiến trúc
from core import config, logging_setup
from models import models          # Qdrant, embedding, Gemini models
from utils import utils            # Utilities chung
from tools import tools            # Reranking, tokenization
from memory import cache           # TTL cache
from retrieval import search, fetch # Tìm kiếm
from services import prompt, render # Xử lý prompt và render
from agents import intent, llm     # Phân tích ý định và LLM
from api import ui                # Gradio interface

# Khởi động Gradio UI
demo = ui.build_ui()
demo.launch()
```

### 2. 📝 Xử lý đầu vào (ui.py → respond_generator)

Khi người dùng nhập câu hỏi:

```python
def respond_generator(message, history_msgs, ...):
    # Validate input
    if not message.strip():
        return fallback_response()
    
    # Bước 1: Phân tích ý định
    intent_info = analyze_intent(message)
```

### 3. 🎯 Phân tích ý định (agents/intent.py)

#### 3.1 Chuẩn hóa câu hỏi (utils/utils.py)
```python
def normalize_legal_query(query):
    # Sửa lỗi chính tả phổ biến
    # Phân loại ý định: true_false, advice, definition, etc.
    # Thêm dấu hỏi hợp ngữ cảnh
    return {
        "normalized_query": text,
        "intent_hint": intent,
        "original_query": original
    }
```

#### 3.2 Phân tích ý định với Gemini
```python
def _intent_via_gemini(query):
    # Gọi Gemini với system prompt chuyên biệt
    # Trả về JSON với schema:
    # - {"intent":"casual","answer":"..."}
    # - {"intent":"legal_answer","normalized_query":"..."}
    # - {"intent":"law_search","filters":{"article_no":int,...}}
```

#### 3.3 Quyết định luồng xử lý
- **casual**: Câu hỏi xã giao → trả lời trực tiếp hoặc stream từ LLM
- **law_search**: Tìm điều luật cụ thể → dùng fetch.py với filters
- **legal_answer**: Câu hỏi pháp lý → dùng search.py với hybrid search

### 4. 🔍 Tìm kiếm tài liệu

#### 4.1 Tìm kiếm theo filter (retrieval/fetch.py)
```python
def _fetch(filters, limit=10):
    # Xây dựng Filter cho Qdrant
    # Tìm kiếm theo article_no, clause_no, point_letter, chapter_number
    # Trả về tài liệu chính xác
```

#### 4.2 Hybrid Search (retrieval/search.py)
```python
async def search_law(query, top_k=15, score_threshold=0.42):
    # Chạy song song 2 tác vụ:
    
    # Task 1: BM25 Search
    async def bm25_search_task():
        # Nếu có filter → tìm docs filtered trước
        # Tokenize và tính BM25 score
        # Trả về top 20 docs với BM25 score
    
    # Task 2: Embedding Search  
    async def embedding_search_task():
        # Encode query thành vector
        # Query Qdrant với vector similarity
        # Trả về top 20 docs với embedding score
    
    # Merge và deduplicate
    # Weighted hybrid scoring (alpha * embedding + beta * BM25)
    # Filter theo threshold
    # BAAI reranking lần 2 (top 7)
```

### 5. 🏗️ Xây dựng Prompt (services/prompt.py)

```python
def build_prompt(query, docs, history_msgs):
    # Thêm lịch sử hội thoại (5 tin nhắn gần nhất)
    # Sắp xếp docs theo độ ưu tiên (ngoại lệ, điều, khoản, điểm)
    # Xây dựng context với format chuẩn:
    #   "Điểm X Khoản Y Điều Z (Chương W) — Tiêu đề: Nội dung"
    
    # Prompt template với:
    # - Vai trò: Trợ lý phân tích pháp luật
    # - Quy tắc: Trích dẫn nguyên văn, kết luận rõ ràng
    # - Format: Trích dẫn → Giải thích → Kết luận
```

### 6. 🤖 Sinh câu trả lời (agents/llm.py)

```python
def stream_answer(prompt, temperature=0.2):
    # Retry mechanism với tenacity
    # Gọi Gemini với streaming
    # Yield từng chunk text
    # Log thời gian first token và tổng thời gian
```

### 7. 🎨 Render và hiển thị (services/render.py)

```python
def docs_to_markdown(docs):
    # Chuyển đổi docs thành markdown
    # Hiển thị: Điều luật + nội dung + điểm số
    
def docs_page_markdown(docs, page, page_size):
    # Phân trang tài liệu
    # Tạo label "Trang X/Y — hiển thị A-B / Tổng"
```

### 8. 💾 Cache và tối ưu (memory/cache.py)

```python
class SimpleTTLCache:
    # TTL cache cho:
    # - embed_cache: 1 giờ, 1024 items
    # - search_cache: 15 phút, 1024 items
```

## Các thành phần công nghệ

### 🤖 AI Models
- **Gemini 2.5 Flash**: Intent analysis và answer generation
- **BGE-M3**: Text embedding cho semantic search
- **BAAI/bge-reranker-base**: Reranking kết quả tìm kiếm

### 🗄️ Vector Database
- **Qdrant**: Lưu trữ embeddings và metadata của điều luật
- **Collection**: Chứa các trường: article_no, clause_no, point_letter, chapter_number, content

### 🔍 Search Methods
1. **BM25**: Keyword-based search với ranking
2. **Embedding**: Semantic search với BGE-M3
3. **Hybrid**: Kết hợp BM25 + Embedding với trọng số
4. **Reranking**: BAAI reranker để tinh chỉnh kết quả

### 🎯 Intent Classification
- **casual**: Câu hỏi xã giao, chào hỏi
- **law_search**: Tìm điều luật cụ thể (có filters)
- **legal_answer**: Câu hỏi pháp lý tổng quát

## Luồng xử lý theo loại câu hỏi

### 📋 Câu hỏi xã giao (casual)
```
Input → Normalize → Intent Analysis → Direct Answer/Stream → Display
```

### 🔍 Tìm điều luật cụ thể (law_search)
```
Input → Normalize → Intent Analysis → Extract Filters → Fetch by Filter → 
Build Prompt → Stream Answer → Display
```

### ⚖️ Câu hỏi pháp lý (legal_answer)
```
Input → Normalize → Intent Analysis → Hybrid Search → 
Merge Results → Weighted Scoring → Rerank → Build Prompt → 
Stream Answer → Display
```

## Tối ưu hóa hiệu suất

### ⚡ Parallel Processing
- BM25 và Embedding search chạy song song
- Async/await cho I/O operations

### 💾 Caching Strategy
- Embedding cache: 1 giờ TTL
- Search cache: 15 phút TTL
- In-memory storage với LRU eviction

### 📊 Monitoring
- Logging chi tiết cho từng bước
- Performance metrics (thời gian, số lượng kết quả)
- Error tracking và fallback mechanisms

## Cấu hình môi trường

```env
# Qdrant Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key
COLLECTION_NAME=hn2014_collection

# AI Models
EMBEDDING_MODEL=BAAI/bge-m3
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL_ID=gemini-2.5-flash

# System Settings
INTENT_DEBUG=0
LOG_LEVEL=INFO
CASUAL_MAX_WORDS=0
INTENT_RAW_PREVIEW_LIMIT=240
```

## Đặc điểm nổi bật

1. **🎯 Intent-Aware**: Phân biệt rõ các loại câu hỏi và xử lý phù hợp
2. **🔍 Multi-Modal Search**: Kết hợp keyword, semantic và reranking
3. **⚡ High Performance**: Parallel processing và intelligent caching
4. **📚 Legal-Accurate**: Trích dẫn chính xác điều luật với format chuẩn
5. **🔄 Streaming Response**: Trải nghiệm người dùng mượt mà
6. **🛡️ Error Resilient**: Fallback mechanisms và comprehensive logging
