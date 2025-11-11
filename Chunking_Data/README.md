
# 📦 Chunking Data Package

> **Hệ thống tái cấu trúc và tối ưu hóa để chunk, embed và upload văn bản luật Việt Nam lên Qdrant vector database**

## 🎯 Tổng quan

Package này cung cấp workflow hoàn chỉnh để xử lý văn bản luật Việt Nam:

1. **📂 File Discovery**: Tìm và catalog các file luật theo category
2. **✂️ Chunking**: Chia văn bản thành chunks theo cấu trúc pháp điển (Chương > Mục > Điều > Khoản > Điểm)
3. **🔄 Merging**: Gộp nhiều chunk files thành 1 file lớn
4. **🧠 Embedding**: Encode chunks thành vectors sử dụng embedding models
5. **📤 Upload**: Upload vectors lên Qdrant vector database

---

## 📁 Cấu trúc Package

```
Chunking_Data/
├── 📦 core/                    # Core modules (thuật toán chính)
│   ├── docx_reader.py          # Đọc file .doc/.docx
│   ├── law_chunker.py          # Chunking logic chính
│   └── law_id_generator.py    # Tự động tạo law_id
│
├── 🔄 pipeline/                # High-level pipelines
│   ├── file_discovery.py      # Tìm và catalog files
│   ├── chunking_pipeline.py   # Batch chunking workflow
│   └── embedding_pipeline.py  # Embedding & upload workflow
│
├── 💾 storage/                 # Data persistence
│   ├── json_handler.py         # Save/load/merge JSON
│   └── qdrant_client.py        # Qdrant operations
│
├── 🔧 scripts/                 # Executable scripts
│   ├── find_files.py           # Tìm files luật
│   ├── chunk_documents.py      # Chunk documents
│   ├── merge_chunks.py         # Merge chunk files
│   └── upload_qdrant.py        # Upload to Qdrant
│
├── ⚙️ config.py                # Configuration
├── 📖 README.md                # Documentation (file này)
└── 🐍 __init__.py              # Package initialization
```

---

## 🚀 Quick Start

### Workflow đầy đủ (3 bước)

```bash
# Bước 1: Tìm tất cả file luật
python -m Chunking_Data.scripts.find_files

# Bước 2: Chunk theo category
python -m Chunking_Data.scripts.chunk_documents --category BDS

# Bước 3: Upload lên Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS/BDS_chunk_*.json \
    --category BDS
```

### Workflow với merge (4 bước)

```bash
# Bước 1: Tìm files
python -m Chunking_Data.scripts.find_files

# Bước 2: Chunk nhiều files
python -m Chunking_Data.scripts.chunk_documents --category BDS

# Bước 3: Merge chunks
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS

# Bước 4: Upload merged file
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS_merged_*.json \
    --category BDS
```

---

## 📚 Chi tiết Scripts

### 1️⃣ find_files.py - Tìm Files Luật

**Mục đích**: Tìm và catalog tất cả file .docx trong `law_content/`

**Usage**:

```bash
# Basic
python -m Chunking_Data.scripts.find_files

# Custom paths
python -m Chunking_Data.scripts.find_files \
    --law-content-dir path/to/laws \
    --output-dir my_data_files \
    --verbose
```

**Output**:

- `data_files/law_file_paths.json` - Tất cả files
- `data_files/BDS/bds_file_paths.json` - Files Bất động sản
- `data_files/DN/dn_file_paths.json` - Files Doanh nghiệp
- `data_files/TM/tm_file_paths.json` - Files Thương mại
- `data_files/QDS/qds_file_paths.json` - Files Quyền dân sự

---

### 2️⃣ chunk_documents.py - Chunk Documents

**Mục đích**: Chunk văn bản luật thành chunks theo cấu trúc pháp điển

**Usage**:

```bash
# Chunk theo category
python -m Chunking_Data.scripts.chunk_documents --category BDS

# Chunk file cụ thể
python -m Chunking_Data.scripts.chunk_documents \
    --file "path/to/law.docx" \
    --law-no "52/2014/QH13" \
    --issued-date "2014-06-19" \
    --effective-date "2015-01-01" \
    --signer "Chủ tịch Quốc hội"

# Chunk tất cả categories
python -m Chunking_Data.scripts.chunk_documents --all

# Chunk với validation
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --validate \
    --verbose
```

**Options**:

- `--category`: BDS, DN, TM, QDS
- `--file`: Chunk 1 file cụ thể
- `--all`: Chunk tất cả
- `--validate`: Validate chunks sau khi tạo
- `--dry-run`: Test mode
- `--verbose`: In log chi tiết

**Output**:

- `data/{CATEGORY}/{NAME}_chunk_HHMMSS_DDMMYY.json`

**Chunk Format**:

```json
{
  "id": "LHNVDG-D5-K2-a",
  "content": "Điều 5. Bảo vệ chế độ hôn nhân và gia đình Khoản 2. ...",
  "metadata": {
    "law_no": "52/2014/QH13",
    "law_title": "Luật Hôn nhân và Gia đình",
    "law_id": "LHNVDG",
    "issued_date": "2014-06-19",
    "effective_date": "2015-01-01",
    "chapter": "Chương I – NHỮNG QUY ĐỊNH CHUNG",
    "chapter_number": 1,
    "article_no": 5,
    "article_title": "Bảo vệ chế độ hôn nhân và gia đình",
    "clause_no": 2,
    "point_letter": "a",
    "exact_citation": "Điều 5 khoản 2 điểm a."
  }
}
```

---

### 3️⃣ merge_chunks.py - Merge Chunk Files

**Mục đích**: Gộp nhiều chunk JSON files thành 1 file lớn

**Usage**:

```bash
# Merge tất cả files trong directory
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS

# Merge files cụ thể
python -m Chunking_Data.scripts.merge_chunks \
    --files file1.json file2.json file3.json \
    --output merged.json

# Merge với pattern
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/ \
    --pattern "BDS_*_chunk_*.json" \
    --category BDS
```

**Options**:

- `--directory`: Thư mục chứa chunk files
- `--files`: List files cụ thể
- `--pattern`: Pattern tìm files (default: `*_chunk_*.json`)
- `--keep-duplicates`: Giữ duplicates (default: remove)
- `--verbose`: In log chi tiết

**Output**:

- `data/{CATEGORY}_merged_chunk_HHMMSS_DDMMYY.json`

---

### 4️⃣ upload_qdrant.py - Upload to Qdrant

**Mục đích**: Embed chunks và upload lên Qdrant vector database

**Usage**:

```bash
# Basic upload
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS_merged.json \
    --category BDS

# Upload với model khác
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --model BAAI/bge-m3

# Force recreate (XÓA data cũ)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --force-recreate

# Dry run (test không upload)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --dry-run
```

**Options**:

- `--model`: Model embedding (default: `minhquan6203/paraphrase-vietnamese-law`)
- `--device`: `cuda` hoặc `cpu` (default: `cuda`)
- `--batch-size`: Batch size (default: 16)
- `--force-recreate`: XÓA data cũ và tạo lại
- `--append`: Append vào collection đã có (default)
- `--dry-run`: Test mode

**Collection Naming**: `{model-name}-{category}`

- Example: `paraphrase-vietnamese-law-BDS`

---

## 🤖 Supported Models

| Model                                                           | Type                  | Dimension | Description                              |
| --------------------------------------------------------------- | --------------------- | --------- | ---------------------------------------- |
| `minhquan6203/paraphrase-vietnamese-law`                        | transformers          | 768       | **Recommended** - Fine-tuned cho luật VN |
| `BAAI/bge-m3`                                                   | transformers          | 1024      | Multilingual model (BAAI)                |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`   | sentence_transformers | 768       | Multilingual SentenceTransformer         |
| `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims` | transformers          | 256       | Vietnamese Law 256-dim                   |

---

## 📂 Categories

| Code  | Full Name    | Description                       |
| ----- | ------------ | --------------------------------- |
| `BDS` | Bất động sản | Luật Bất động sản, Nhà ở, Đất đai |
| `DN`  | Doanh nghiệp | Luật Doanh nghiệp, Công ty        |
| `TM`  | Thương mại   | Luật Thương mại                   |
| `QDS` | Quyền dân sự | Luật Dân sự, Hôn nhân Gia đình    |

---

## ⚙️ Configuration

### Environment Variables

Tạo file `.env` trong root directory:

```bash
# Qdrant Configuration (REQUIRED)
QDRANT_URL=https://your-qdrant-url.com
QDRANT_API_KEY=your-api-key-here

# Optional
EMBEDDING_MODEL=minhquan6203/paraphrase-vietnamese-law
DEVICE=cuda
BATCH_SIZE=16
```

### Kiểm tra Config

```bash
python -m Chunking_Data.config
```

---

## 🔧 Sử dụng như Library

### Example 1: Chunk một file

```python
from Chunking_Data.pipeline.chunking_pipeline import ChunkingPipeline

pipeline = ChunkingPipeline(verbose=True)

chunks, stats = pipeline.process_single_file(
    file_path="path/to/law.docx",
    law_id="LHNVDG",
    law_no="52/2014/QH13",
    law_title="Luật Hôn nhân và Gia đình"
)

print(f"Created {len(chunks)} chunks")
print(f"Stats: {stats}")
```

### Example 2: Upload lên Qdrant

```python
from Chunking_Data.pipeline.embedding_pipeline import EmbeddingPipeline
from Chunking_Data.storage.json_handler import load_chunks_from_json

# Load chunks
chunks = load_chunks_from_json("data/BDS_merged.json")

# Create pipeline
pipeline = EmbeddingPipeline(
    model_name="minhquan6203/paraphrase-vietnamese-law",
    device="cuda",
    verbose=True
)

# Upload
results = pipeline.process_and_upload(
    chunks=chunks,
    category="BDS",
    append_mode=True
)

print(f"Uploaded to: {results['collection_name']}")
print(f"Total vectors: {results['total_vectors']}")
```

### Example 3: Merge chunks

```python
from Chunking_Data.storage.json_handler import (
    merge_chunk_files,
    save_chunks_to_json
)

# Merge
merged_chunks = merge_chunk_files(
    file_paths=["file1.json", "file2.json", "file3.json"],
    remove_duplicates=True,
    verbose=True
)

# Save
save_chunks_to_json(merged_chunks, "merged_output.json")
=======
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
>>>>>>> 985580bf68add13b2bc7f26f77585e9417bff953
```

---

<<<<<<< HEAD
## 📊 Chunking Logic

### Cấu trúc pháp điển

```
Chương (Chapter)
  └── Mục (Section) [optional]
      └── Điều (Article)
          ├── Intro text [optional]
          └── Khoản (Clause)
              ├── Intro text [optional]
              └── Điểm (Point)
```

### Features

✅ **2-pass parsing**: Pre-scan + strict parsing
✅ **Clause intro injection**: Tiêm intro khoản vào mỗi điểm
✅ **Article intro handling**: Xử lý intro điều thông minh
✅ **Auto law_id generation**: Tự động tạo ID từ tên file
✅ **Citation generation**: Tạo exact_citation cho mỗi chunk
✅ **Statistics tracking**: Thu thập stats đầy đủ

### Regex Patterns

- **Article**: `^Điều\s+(\d+)`
- **Chapter**: `^Chương\s+([IVXLCDM]+|\d+)`
- **Section**: `^Mục\s+(\d+)`
- **Clause**: `^\s*(\d+)\.`
- **Point**: `^\s*([a-zA-ZđĐ])[\)\.]\s+`

---

## 🐛 Troubleshooting

### Lỗi: "QDRANT_URL not set"

**Giải pháp**: Tạo file `.env` và set `QDRANT_URL`

```bash
echo "QDRANT_URL=https://your-url.com" > .env
echo "QDRANT_API_KEY=your-key" >> .env
```

### Lỗi: "Vector size mismatch"

**Nguyên nhân**: Đang append vào collection có vector dimension khác

**Giải pháp**: Sử dụng `--force-recreate` để tạo lại collection

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --force-recreate
```

### Lỗi: "CUDA out of memory"

**Giải pháp 1**: Giảm batch size

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --batch-size 8
```

**Giải pháp 2**: Sử dụng CPU

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --device cpu
```

---

## 📝 Notes

### Law ID Generation

Package tự động tạo law_id thông minh từ tên file:

- **Luật gốc**: `Luật Sở hữu trí tuệ.docx` → `LSHTT`
- **Luật sửa đổi**: `Luật sửa đổi Luật SHTT.docx` → `LSĐBSLSHTT`
- **Văn bản hợp nhất**: `VBHN Luật Xây dựng.docx` → `LXAYDUNG`

### Upload Modes

| Mode                 | Mô tả                  | Khi nào dùng         |
| -------------------- | ---------------------- | -------------------- |
| **Append** (default) | Thêm vào collection    | Upload thêm data mới |
| **Force Recreate**   | XÓA data cũ và tạo lại | Làm lại từ đầu       |

---

## 🎯 Best Practices

### 1. Workflow khuyến nghị

```bash
# Always start with find_files
python -m Chunking_Data.scripts.find_files

# Chunk by category (easier to manage)
python -m Chunking_Data.scripts.chunk_documents --category BDS --validate

# Merge if multiple files
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS

# Upload with default recommended model
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS_merged.json \
    --category BDS
```

### 2. Organization

- ✅ Chunk theo category riêng biệt
- ✅ Merge trước khi upload (dễ quản lý hơn)
- ✅ Sử dụng `--validate` để check quality
- ✅ Backup chunks JSON files

### 3. Performance

- ✅ Sử dụng GPU nếu có (`--device cuda`)
- ✅ Batch size phù hợp với VRAM (16 cho 8GB)
- ✅ Merge chunks trước khi upload (giảm overhead)

---

## 🤝 Contributing

Package được tái cấu trúc từ `vn-law-embedding-benchmark` với mục đích:

- ✨ **Modular**: Tách biệt core logic, pipelines, storage
- ✨ **Reusable**: Dễ dàng import và sử dụng
- ✨ **Maintainable**: Code clean, well-documented
- ✨ **User-friendly**: Scripts đơn giản, clear instructions

---

## 📜 License

MIT License

---

## 📧 Support

Nếu có vấn đề hoặc câu hỏi, vui lòng:

1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Xem lại [Configuration](#%EF%B8%8F-configuration)
3. Đọc kỹ error messages (thường có hint)

---

**Made with ❤️ for Vietnamese Legal AI**
=======
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
