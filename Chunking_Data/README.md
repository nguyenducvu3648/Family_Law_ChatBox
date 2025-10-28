# Chunking Data - Vietnamese Legal Document Processing

Hệ thống chunk, embed và upload văn bản luật Việt Nam lên Qdrant vector database.

---

## Quick Start

```bash
# 1. Tìm files luật
python -m Chunking_Data.scripts.find_files (đảm bảo khó foder law_contents)

# 2. Chunk văn bản
python -m Chunking_Data.scripts.chunk_documents --category BDS

# 3. Merge chunks
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS

# 4. Upload lên Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged_chunk_*.json" \
    --category BDS

# 5. Reindex collections (optional - tối ưu search performance)
python -m Chunking_Data.scripts.reindex_qdrant_collections --all
```

---

## Project Structure

```
Chunking_Data/
│
├── core/              # Core algorithms (docx_reader, law_chunker, law_id_generator)
├── pipeline/          # Orchestration workflows (file_discovery, chunking_pipeline, embedding_pipeline)
├── storage/           # Data persistence (json_handler, qdrant_client)
├── evaluation/        # AI Review with Gemini (ai_reviewer)
└── scripts/           # CLI tools (find_files, chunk_documents, merge_chunks, upload_qdrant, reindex_qdrant_collections)
```

### Documentation

| Module         | Description                            | Documentation                                |
| -------------- | -------------------------------------- | -------------------------------------------- |
| **core**       | Pure algorithms for chunking           | [core/README.md](core/README.md)             |
| **pipeline**   | High-level workflows and orchestration | [pipeline/README.md](pipeline/README.md)     |
| **storage**    | JSON and Qdrant operations             | [storage/README.md](storage/README.md)       |
| **evaluation** | AI-powered quality assurance           | [evaluation/README.md](evaluation/README.md) |
| **scripts**    | CLI tools (chunk, upload, reindex)     | [scripts/README.md](scripts/README.md)       |

---

## Workflows

### Full Pipeline

```bash
# Step 1: Discover files
python -m Chunking_Data.scripts.find_files

# Step 2: Chunk by category
python -m Chunking_Data.scripts.chunk_documents --category BDS --validate

# Step 3: Merge chunks
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS

# Step 4: Upload to Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged_chunk_*.json" \
    --category BDS

# Step 5: Reindex collections (tối ưu search performance)
python -m Chunking_Data.scripts.reindex_qdrant_collections --all
```

### Single File

```bash
# Chunk
python -m Chunking_Data.scripts.chunk_documents \
    --file "law_content/path/to/law.docx" \
    --law-no "52/2014/QH13"

# Upload
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/law_chunk_*.json" \
    --category BDS

# Reindex (nếu cần tối ưu search)
python -m Chunking_Data.scripts.reindex_qdrant_collections --collection your-collection-name
```

### AI Review (Gemini) 🤖

Tự động kiểm tra chất lượng chunks bằng Gemini AI:

```bash
# Quick Start
echo "GEMINI_API_KEY=your-api-key" > .env
python -m Chunking_Data.scripts.chunk_documents --category BDS --AI

# Production mode (strict)
python -m Chunking_Data.scripts.chunk_documents --category BDS --AI --strict-ok-only
```

**Features:** Phát hiện lỗi chunking, metadata validation, confidence scoring

**Docs:** [evaluation/README.md](evaluation/README.md) - Chi tiết AI Review

### Reindex Qdrant Collections 🔍

Tạo/cập nhật payload indexes cho collections Qdrant hiện có để tối ưu hóa search performance:

```bash
# Reindex tất cả collections
python -m Chunking_Data.scripts.reindex_qdrant_collections --all

# Reindex collection cụ thể
python -m Chunking_Data.scripts.reindex_qdrant_collections --collection BAAI-bge-m3-LHN2014

# Reindex nhiều collections
python -m Chunking_Data.scripts.reindex_qdrant_collections \
    --collection BAAI-bge-m3-BDS \
    --collection hybrid-QDS

# Test mode (không thực hiện thay đổi)
python -m Chunking_Data.scripts.reindex_qdrant_collections --all --dry-run
```

**Features:**

- Index tự động cho tất cả metadata fields pháp điển
- Hỗ trợ DateTime, Integer, Keyword, Text indexing
- Skip indexes đã tồn tại
- Báo cáo chi tiết kết quả reindexing

**Indexed Fields:**

- Thông tin luật: `law_id`, `law_title`, `law_no`
- Thời gian: `issued_date`, `effective_date`, `expiry_date`
- Cấu trúc pháp điển: `chapter`, `article_no`, `clause_no`, `point_letter`, etc.

### Programmatic Usage

```python
from Chunking_Data.pipeline.chunking_pipeline import ChunkingPipeline
from Chunking_Data.storage.json_handler import save_chunks_to_json

pipeline = ChunkingPipeline(verbose=True)
chunks, stats = pipeline.process_single_file("law.docx")
save_chunks_to_json(chunks, "output.json")
```

---

## Supported Models

| Model                                    | Dimension | Type         |
| ---------------------------------------- | --------- | ------------ |
| `minhquan6203/paraphrase-vietnamese-law` | 768       | transformers |
| `BAAI/bge-m3`                            | 1024      | transformers |

---

## Architecture

```
Scripts → Pipeline → Core
        ↘ Storage ↗
```

---

### Command Help

```bash
python -m Chunking_Data.scripts.find_files --help
python -m Chunking_Data.scripts.chunk_documents --help
python -m Chunking_Data.scripts.merge_chunks --help
python -m Chunking_Data.scripts.upload_qdrant --help
python -m Chunking_Data.scripts.reindex_qdrant_collections --help
python -m Chunking_Data --validate-config  # Config validation
```
