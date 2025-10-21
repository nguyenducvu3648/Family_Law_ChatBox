# 🏗️ Chunking_Data Architecture

## 📐 Design Principles

Package được thiết kế theo các nguyên tắc:

1. **Separation of Concerns**: Tách biệt rõ ràng giữa core logic, pipelines, storage
2. **Modularity**: Mỗi module có trách nhiệm cụ thể, dễ test và maintain
3. **Reusability**: Code có thể reuse ở nhiều nơi (scripts, notebooks, API)
4. **Progressive Enhancement**: Từ simple đến complex (core → pipeline → scripts)

---

## 📦 Module Layers

```
┌─────────────────────────────────────────┐
│         🔧 Scripts Layer                │  ← User-facing CLI
│  (find_files, chunk_documents, ...)    │
├─────────────────────────────────────────┤
│         🔄 Pipeline Layer               │  ← High-level workflows
│  (ChunkingPipeline, EmbeddingPipeline) │
├─────────────────────────────────────────┤
│         💾 Storage Layer                │  ← Data persistence
│  (json_handler, qdrant_client)         │
├─────────────────────────────────────────┤
│         📦 Core Layer                   │  ← Pure algorithms
│  (docx_reader, law_chunker, ...)       │
└─────────────────────────────────────────┘
```

---

## 🔍 Core Layer (Thuật toán thuần túy)

### docx_reader.py

- **Mục đích**: Đọc file .doc/.docx
- **Methods**: 5 fallback methods (python-docx → docx2txt → textract → pypandoc → antiword)
- **Input**: File path
- **Output**: Raw text string
- **Dependencies**: Không phụ thuộc module khác

### law_chunker.py

- **Mục đích**: Chunking văn bản luật
- **Algorithm**: 2-pass parsing (pre-scan + strict parsing)
- **Features**:
  - State machine cho parsing
  - Clause intro injection
  - Article intro handling
  - Citation generation
- **Input**: Raw text + metadata
- **Output**: (chunks, stats)
- **Dependencies**: Chỉ dùng stdlib (re, unicodedata)

### law_id_generator.py

- **Mục đích**: Tự động tạo law_id
- **Logic**:
  1. Thử mapping với LAW_MAPPINGS dictionary
  2. Fallback: Tạo từ chữ cái đầu
  3. Xử lý luật sửa đổi/bổ sung
- **Input**: File name
- **Output**: Law ID string
- **Dependencies**: Chỉ dùng re, unicodedata

---

## 🔄 Pipeline Layer (Workflows)

### file_discovery.py

- **Mục đích**: Tìm và catalog files
- **Functions**:
  - `find_law_files()`: Tìm .docx files
  - `find_question_files()`: Tìm .xlsx files
  - `create_file_paths_list()`: Tạo danh sách paths
- **Input**: Directory path
- **Output**: File paths list + category mapping
- **Dependencies**: glob, os

### chunking_pipeline.py

- **Mục đích**: Batch chunking workflow
- **Class**: `ChunkingPipeline`
- **Methods**:
  - `process_single_file()`: Chunk 1 file
  - `process_files()`: Chunk nhiều files (batch)
  - `validate_chunks()`: Validate quality
- **Features**:
  - Progress tracking
  - Error handling và retry
  - Statistics collection
- **Dependencies**: core.docx_reader, core.law_chunker, core.law_id_generator

### embedding_pipeline.py

- **Mục đích**: Embedding & upload workflow
- **Class**: `EmbeddingPipeline`
- **Methods**:
  - `extract_texts()`: Extract texts từ chunks
  - `encode_texts()`: Encode thành vectors
  - `upload_to_qdrant()`: Upload lên Qdrant
  - `process_and_upload()`: End-to-end
- **Features**:
  - Lazy model loading
  - Dimension auto-detection
  - Collection management
- **Dependencies**: storage.qdrant_client

---

## 💾 Storage Layer (Data Persistence)

### json_handler.py

- **Mục đích**: JSON operations
- **Functions**:
  - `save_chunks_to_json()`: Lưu chunks
  - `load_chunks_from_json()`: Đọc chunks
  - `merge_chunk_files()`: Merge nhiều files
  - `get_file_stats()`: Lấy statistics
- **Features**:
  - Auto create directories
  - Duplicate removal
  - Validation
- **Dependencies**: json, pathlib

### qdrant_client.py

- **Mục đích**: Qdrant operations
- **Functions**:
  - `get_qdrant_client()`: Kết nối
  - `ensure_collection()`: Tạo collection
  - `ensure_or_append_collection()`: Create/append
  - `upsert_embeddings_to_qdrant()`: Upload
  - `encode_texts()`: Embedding
  - `get_embedding_dimension()`: Get dimension
- **Features**:
  - Connection pooling
  - Batch upload
  - Payload indexing
  - Model loading (transformers/sentence-transformers)
- **Dependencies**: qdrant-client, torch, transformers

---

## 🔧 Scripts Layer (User Interface)

Scripts là thin wrappers around pipelines, cung cấp CLI interface.

### find_files.py

```python
Pipeline: file_discovery.find_law_files()
Output: JSON files (law_file_paths.json, category_file_paths.json)
```

### chunk_documents.py

```python
Pipeline: ChunkingPipeline.process_files()
Output: Chunk JSON files (data/{category}/*.json)
```

### merge_chunks.py

```python
Pipeline: json_handler.merge_chunk_files()
Output: Merged JSON file (data/{category}_merged_*.json)
```

### upload_qdrant.py

```python
Pipeline: EmbeddingPipeline.process_and_upload()
Output: Qdrant collection ({model}-{category})
```

---

## 🔀 Data Flow

### Workflow 1: Chunking

```
law_content/               (Raw files)
    ↓
find_files.py
    ↓
data_files/*.json         (File paths catalog)
    ↓
chunk_documents.py
    ↓
data/{category}/*.json    (Individual chunks)
```

### Workflow 2: Merging

```
data/{category}/*.json    (Individual chunks)
    ↓
merge_chunks.py
    ↓
data/{category}_merged.json  (Merged chunks)
```

### Workflow 3: Upload

```
data/*.json               (Chunks)
    ↓
upload_qdrant.py
    ↓
Qdrant Collection         (Vectors)
```

---

## 🎯 Design Decisions

### 1. Tại sao tách core từ pipeline?

**Lý do**:

- Core modules là pure functions → dễ test
- Pipelines thêm error handling, logging, progress
- Có thể reuse core ở nhiều contexts (CLI, API, notebooks)

### 2. Tại sao không dùng classes cho core?

**Lý do**:

- Functions đơn giản hơn classes cho pure logic
- Không có state cần maintain
- Dễ compose và test

### 3. Tại sao tách storage layer riêng?

**Lý do**:

- Abstraction cho data persistence
- Dễ swap implementation (JSON → database)
- Single responsibility

### 4. Tại sao scripts là thin wrappers?

**Lý do**:

- Logic nằm ở pipelines → dễ test
- Scripts chỉ lo CLI interface
- Có thể dùng pipelines trực tiếp trong code

---

## 📊 Complexity Analysis

### Module Complexity (Lines of Code)

```
core/
  ├── docx_reader.py        ~150 LOC   [Low complexity]
  ├── law_chunker.py        ~650 LOC   [High complexity - core algorithm]
  └── law_id_generator.py   ~200 LOC   [Medium complexity]

pipeline/
  ├── file_discovery.py     ~300 LOC   [Medium complexity]
  ├── chunking_pipeline.py  ~350 LOC   [Medium complexity]
  └── embedding_pipeline.py ~300 LOC   [Medium complexity]

storage/
  ├── json_handler.py       ~200 LOC   [Low complexity]
  └── qdrant_client.py      ~450 LOC   [Medium complexity]

scripts/
  ├── find_files.py         ~120 LOC   [Low complexity]
  ├── chunk_documents.py    ~200 LOC   [Low complexity]
  ├── merge_chunks.py       ~150 LOC   [Low complexity]
  └── upload_qdrant.py      ~170 LOC   [Low complexity]
```

### Dependency Graph

```
Scripts
  ↓ depends on
Pipelines
  ↓ depends on
Storage + Core
  ↓ depends on
External libs (qdrant-client, transformers, etc.)
```

---

## 🧪 Testing Strategy

### Unit Tests (Per module)

```python
# test_core/
test_docx_reader.py      # Test reading different formats
test_law_chunker.py      # Test parsing logic
test_law_id_generator.py # Test ID generation

# test_pipeline/
test_chunking_pipeline.py    # Test batch processing
test_embedding_pipeline.py   # Test encoding & upload

# test_storage/
test_json_handler.py     # Test save/load/merge
test_qdrant_client.py    # Test Qdrant operations
```

### Integration Tests

```python
test_e2e_chunking.py     # End-to-end chunking workflow
test_e2e_upload.py       # End-to-end upload workflow
```

---

## 🚀 Performance Considerations

### 1. Memory Management

- **Core chunking**: Streaming processing (không load toàn bộ vào memory)
- **Embedding**: Batch processing với clear cache
- **Upload**: Batch upsert (100 vectors/batch)

### 2. GPU Optimization

- **Lazy model loading**: Chỉ load khi cần
- **Batch size tuning**: 16 cho 8GB VRAM
- **Memory cleanup**: `torch.cuda.empty_cache()` sau mỗi batch

### 3. I/O Optimization

- **JSON**: Streaming write cho large files
- **Qdrant**: Batch upload thay vì single
- **File discovery**: Glob với recursive=True

---

## 🔮 Future Enhancements

### Potential Improvements

1. **Async I/O**: Sử dụng asyncio cho file operations
2. **Caching**: Cache embeddings để tránh re-encode
3. **Parallel Processing**: Multi-processing cho batch chunking
4. **Progress Persistence**: Save progress để resume
5. **Web API**: Flask/FastAPI wrapper cho pipelines

### Extensibility Points

- **New readers**: Thêm support cho PDF, TXT
- **New chunkers**: Custom chunking strategies
- **New storages**: PostgreSQL, MongoDB
- **New models**: Easy to add new embedding models

---

**Architecture Version**: 2.0.0  
**Last Updated**: 2025-01-14
