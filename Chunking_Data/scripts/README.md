# 🔧 Scripts - CLI Tools

> **Vai trò**: Command-line tools để chạy các workflows

---

## Scripts là **user-facing CLI tools** để:

```
✅ Chạy workflows bằng command line
✅ Không cần viết code
✅ Flexible với arguments
✅ Automation friendly
```

**Scripts gọi Pipeline** (không gọi Core trực tiếp):

```
User → Scripts → Pipeline → Core
```

---

## 📁 Files Trong Scripts

### 1. `find_files.py` - Tìm Files Luật

**Chức năng**: Scan `law_content/` và tạo JSON catalog

**Usage**:

```bash
python -m Chunking_Data.scripts.find_files
```

**Arguments**:

```bash
--law-content-dir PATH    # Thư mục chứa luật (default: law_content)
--output-dir PATH         # Output directory (default: data_files)
--verbose, -v             # Log chi tiết
```

**Output**:

```
data_files/
├── law_file_paths.json         # Tất cả files
├── BDS/
│   └── bds_file_paths.json     # Bất động sản
├── DN/
│   └── dn_file_paths.json      # Doanh nghiệp
├── TM/
│   └── tm_file_paths.json      # Thương mại
└── QDS/
    └── qds_file_paths.json     # Quyền dân sự
```

**Example**:

```bash
# Basic
python -m Chunking_Data.scripts.find_files

# Custom paths
python -m Chunking_Data.scripts.find_files \
    --law-content-dir "my_laws" \
    --output-dir "my_output" \
    --verbose
```

---

### 2. `chunk_documents.py` - Chunk Văn Bản

**Chức năng**: Chunk văn bản luật thành JSON chunks

**Usage**:

```bash
python -m Chunking_Data.scripts.chunk_documents [OPTIONS]
```

**Arguments**:

#### Input (chọn 1 trong 3):

```bash
--category BDS         # Chunk theo category
--file PATH            # Chunk 1 file cụ thể
--all                  # Chunk tất cả categories
```

#### Metadata:

```bash
--law-no "52/2014/QH13"           # Số hiệu luật
--issued-date "2014-06-19"        # Ngày ban hành (YYYY-MM-DD)
--effective-date "2015-01-01"     # Ngày hiệu lực
--signer "Chủ tịch Quốc hội"      # Người ký
```

#### Options:

```bash
--output-dir DIR       # Output directory (default: data)
--output-name NAME     # Custom output filename
--validate             # Validate chunks sau khi tạo
--dry-run              # Test mode, không ghi file
--verbose, -v          # Log chi tiết
```

**Examples**:

```bash
# 1. Chunk theo category
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --validate \
    --verbose

# 2. Chunk 1 file với metadata đầy đủ
python -m Chunking_Data.scripts.chunk_documents \
    --file "law_content/Bất động sản/Luật Nhà ở/Luật Nhà ở 2023.docx" \
    --law-no "65/2023/QH15" \
    --issued-date "2023-06-19" \
    --effective-date "2024-01-01" \
    --signer "Chủ tịch Quốc hội"

# 3. Chunk tất cả
python -m Chunking_Data.scripts.chunk_documents --all --validate

# 4. Dry run (test không ghi file)
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --dry-run
```

**Output**:

```
data/BDS/law1_chunk_143052_231025.json
data/BDS/law2_chunk_143103_231025.json
...
```

---

### 3. `merge_chunks.py` - Merge Chunk Files

**Chức năng**: Gộp nhiều JSON files thành 1 file lớn

**Usage**:

```bash
python -m Chunking_Data.scripts.merge_chunks [OPTIONS]
```

**Arguments**:

#### Input (chọn 1):

```bash
--directory DIR        # Merge tất cả files trong directory
--files FILE1 FILE2    # Merge files cụ thể
```

#### Options:

```bash
--pattern PATTERN      # Pattern tìm files (default: *_chunk_*.json)
--category CAT         # Category cho output filename
--output PATH          # Output file path (custom)
--output-dir DIR       # Output directory (default: data)
--keep-duplicates      # Giữ duplicates (default: remove by ID)
--dry-run              # Test mode
```

**Examples**:

```bash
# 1. Merge tất cả trong directory
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS

# 2. Merge với pattern
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --pattern "luat_*_chunk_*.json" \
    --category BDS

# 3. Merge files cụ thể
python -m Chunking_Data.scripts.merge_chunks \
    --files data/BDS/law1.json data/BDS/law2.json \
    --output data/BDS_custom.json

# 4. Dry run
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS \
    --dry-run
```

**Output**:

```
data/BDS_merged_chunk_HHMMSS_DDMMYY.json
```

**Stats**:

```
Total chunks: 1,234
Duplicates removed: 15
Law IDs: LNHAO, LDATDAI, LKBDS
File size: 5.6 MB
```

---

### 4. `upload_qdrant.py` - Upload Lên Qdrant

**Chức năng**: Embed chunks và upload lên Qdrant vector database

**Usage**:

```bash
python -m Chunking_Data.scripts.upload_qdrant [OPTIONS]
```

**Arguments**:

#### Required:

```bash
--chunk-file PATH      # Path to chunk JSON file
--category CAT         # Category name (VD: BDS, DN)
```

#### Model Options:

```bash
--model MODEL_NAME     # Embedding model
                       # Default: minhquan6203/paraphrase-vietnamese-law
--device cuda|cpu      # Device (default: cuda)
--batch-size N         # Batch size (default: 16)
```

#### Collection Options:

```bash
--force-recreate       # XÓA data cũ và tạo lại collection
--append               # Append vào collection (DEFAULT)
```

#### Other:

```bash
--dry-run              # Test mode, không upload
--verbose, -v          # Log chi tiết
```

**Examples**:

```bash
# 1. Upload với default model
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged.json" \
    --category BDS

# 2. Upload với model khác
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS.json" \
    --category BDS \
    --model "BAAI/bge-m3" \
    --device cuda \
    --batch-size 16

# 3. Force recreate (XÓA data cũ)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS.json" \
    --category BDS \
    --force-recreate

# 4. Dùng CPU (nếu không có GPU)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS.json" \
    --category BDS \
    --device cpu \
    --batch-size 8

# 5. Dry run (test không upload)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS.json" \
    --category BDS \
    --dry-run
```

**Collection Naming**:

```
{model-name}-{category}

Examples:
- paraphrase-vietnamese-law-BDS
- BAAI-bge-m3-DN
```

**Output**:

```
✅ SUCCESS!
   Collection: paraphrase-vietnamese-law-BDS
   Vectors: 1,234
   Dimension: 768
```

---

## 🎯 Workflow Khuyến Nghị

### Workflow Đầy Đủ (4 Bước):

```bash
# Bước 1: Tìm files (chạy 1 lần)
python -m Chunking_Data.scripts.find_files

# Bước 2: Chunk theo category
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --validate

# Bước 3: Merge chunks
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS

# Bước 4: Upload lên Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged_chunk_*.json" \
    --category BDS
```

### Workflow Nhanh (Cho 1 File):

```bash
# Chunk 1 file
python -m Chunking_Data.scripts.chunk_documents \
    --file "law.docx"

# Upload luôn (không cần merge)
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/law_chunk_*.json" \
    --category BDS
```
