# 🔄 Pipeline - Orchestration Layer

> **Vai trò**: Điều phối các core functions + Xử lý lỗi + Batch processing

---

## Pipeline là **tầng trung gian** giữa:

- **Core** (thuật toán thuần túy)
- **Scripts** (CLI tools)

```
Scripts → Pipeline → Core
```

**Pipeline thêm gì so với Core?**

```
✅ Loop nhiều files (batch processing)
✅ Error handling (file not found, read error, ...)
✅ Progress tracking (log tiến trình)
✅ Aggregate statistics (tổng hợp kết quả)
✅ Auto law_id generation
```

---

## 📁 Files Trong Pipeline

### 1. `file_discovery.py` - Tìm Files Luật

**Chức năng**: Scan `law_content/` và tạo catalog JSON

**Main Functions**:

#### `find_law_files(law_content_dir, verbose=True)`

```python
→ (all_files: List[str], files_by_category: Dict)
```

**Logic**:

1. Tìm tất cả `.docx` trong `law_content/` (recursive)
2. **Lọc** chỉ lấy files trong folder có keywords:
   - "Luật\_"
   - "văn bản pháp luật"
   - "Văn bản pháp lý"
   - "văn bản quy phạm pháp luật"
3. **Phân loại** theo category dựa vào path:
   - `Bất động sản/` → BDS
   - `Doanh nghiệp_/` → DN
   - `Luật Thương Mại/` → TM
   - `Quyền dân sự_/` → QDS

**Example**:

```python
from Chunking_Data.pipeline.file_discovery import find_law_files

all_files, by_category = find_law_files("law_content", verbose=True)

print(f"Tổng: {len(all_files)} files")
print(f"BDS: {len(by_category['BDS'])} files")
print(f"DN: {len(by_category['DN'])} files")
```

#### `create_file_paths_list(law_content_dir)`

```python
→ List[Dict] với format:
[
  {
    'path': 'law_content/Bất động sản/Luật Nhà ở/...',
    'relative_path': 'Bất động sản/Luật Nhà ở/...',
    'category': 'BDS',
    'file_name': 'Luật Nhà ở 2023.docx',
    'extension': '.docx'
  },
  ...
]
```

#### `create_category_file_paths(files_by_category, output_dir)`

```python
Lưu JSON files:
- data_files/law_file_paths.json (tất cả)
- data_files/BDS/bds_file_paths.json
- data_files/DN/dn_file_paths.json
- data_files/TM/tm_file_paths.json
- data_files/QDS/qds_file_paths.json
```

**Khi nào dùng**: Script `find_files.py` gọi functions này

---

### 2. `chunking_pipeline.py` - Class ChunkingPipeline

**Chức năng**: Orchestrate việc chunk nhiều files + Error handling

**Class Definition**:

```python
class ChunkingPipeline:
    def __init__(self, verbose: bool = True):
        """
        Setup pipeline

        Args:
            verbose: In log chi tiết
        """

    def process_single_file(
        self,
        file_path: str,
        law_no: str = "",
        law_title: Optional[str] = None,
        law_id: Optional[str] = None,
        issued_date: str = "",
        effective_date: str = "",
        expiry_date: Optional[str] = None,
        signer: str = ""
    ) → (chunks: List[Dict], stats: Dict):
        """
        Chunk 1 file

        Returns:
            (chunks, stats)

        Raises:
            FileNotFoundError: File không tồn tại
            ValueError: Content quá ngắn
        """

    def process_files(
        self,
        file_paths: List[Dict],
        default_law_no: str = "",
        default_issued_date: str = "",
        default_effective_date: str = "",
        default_signer: str = ""
    ) → (all_chunks: List[Dict], summary: Dict):
        """
        Chunk nhiều files (batch)

        Returns:
            (all_chunks, summary)
        """
```

**Workflow của `process_single_file()`**:

```
Input: file_path
   ↓
1. Check file tồn tại
   ↓
2. Đọc file (gọi core.docx_reader.read_docx)
   ↓ [Error handling: cannot read]
3. Validate content (length > 100 chars)
   ↓ [Error handling: too short]
4. Tạo law_id (gọi core.law_id_generator)
   ↓
5. Chunk (gọi core.law_chunker.chunk_law_document)
   ↓
Output: (chunks, stats)
```

**Workflow của `process_files()`**:

```
Input: List[file_info]
   ↓
Loop:
  ├─ Try:
  │   ├─ process_single_file(file)
  │   ├─ Collect chunks
  │   └─ Aggregate stats
  └─ Except:
      ├─ Log error
      ├─ Warnings.append(error)
      └─ Continue (không dừng)
   ↓
Output: (all_chunks, summary_stats)
```

**Stats Aggregation**:

```python
summary = {
    'chapters_seen': [...],      # Unique chapters
    'articles': 133,             # Tổng số điều
    'article_intro': 39,         # Tổng intro điều
    'clauses': 274,              # Tổng khoản
    'points': 80,                # Tổng điểm
    'citations': [...],          # All citations
    'warnings': [...],           # Errors occurred
    'total_chunks': 393,
    'successful_files': 8,
    'failed_files': 2
}
```

**Example Usage**:

```python
from Chunking_Data.pipeline.chunking_pipeline import ChunkingPipeline

# Case 1: Chunk 1 file
pipeline = ChunkingPipeline(verbose=True)
chunks, stats = pipeline.process_single_file(
    file_path="law.docx",
    law_no="52/2014/QH13",
    issued_date="2014-06-19"
)
print(f"Created {len(chunks)} chunks")

# Case 2: Chunk nhiều files
file_paths = [
    {'path': 'law1.docx', 'file_name': 'law1.docx', 'category': 'BDS'},
    {'path': 'law2.docx', 'file_name': 'law2.docx', 'category': 'BDS'},
]

all_chunks, summary = pipeline.process_files(
    file_paths=file_paths,
    default_law_no="52/2014/QH13"
)

print(f"Total: {summary['total_chunks']} chunks")
print(f"Success: {summary['successful_files']}/{len(file_paths)} files")
```

---

### 3. `embedding_pipeline.py` - Class EmbeddingPipeline

**Chức năng**: End-to-end embed chunks + upload lên Qdrant

**Class Definition**:

```python
class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "minhquan6203/paraphrase-vietnamese-law",
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Setup embedding pipeline

        Args:
            model_name: Embedding model
            device: 'cuda' hoặc 'cpu'
            verbose: Log chi tiết
        """

    def extract_texts(
        self,
        chunks: List[Dict]
    ) → List[str]:
        """Extract content từ chunks"""

    def encode_texts(
        self,
        texts: List[str]
    ) → np.ndarray:
        """
        Encode texts thành embeddings
        (Lazy load model lần đầu gọi)
        """

    def upload_to_qdrant(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict],
        collection_name: str,
        append_mode: bool = True
    ):
        """Upload embeddings + metadata lên Qdrant"""

    def process_and_upload(
        self,
        chunks: List[Dict],
        category: str,
        append_mode: bool = True
    ) → Dict:
        """
        END-TO-END: Extract → Encode → Upload

        Returns:
            {
                'collection_name': 'paraphrase-vietnamese-law-BDS',
                'total_vectors': 393,
                'vector_size': 768
            }
        """
```

**Workflow `process_and_upload()`**:

```
Input: chunks + category
   ↓
1. Extract texts từ chunks
   ↓
2. Lazy load model (chỉ load lần đầu)
   ↓
3. Encode texts → embeddings
   (Batch processing: 16 texts/batch)
   ↓
4. Connect Qdrant
   ↓
5. Create hoặc Append collection
   ↓
6. Upload embeddings + metadata
   (Batch: 100 vectors/batch)
   ↓
Output: collection info
```

**Lazy Model Loading**:

```python
# Lần đầu gọi encode_texts():
if self.model is None:
    self.model = load_model(...)  # Load model

# Lần sau: Dùng model đã load
embeddings = self.model.encode(texts)
```

**Collection Naming**:

```python
collection_name = f"{clean_model_name}-{category}"
# Example: "paraphrase-vietnamese-law-BDS"
```

**Example Usage**:

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

# Process & Upload (all in one!)
results = pipeline.process_and_upload(
    chunks=chunks,
    category="BDS",
    append_mode=True
)

print(f"✅ Uploaded to: {results['collection_name']}")
print(f"   Total vectors: {results['total_vectors']}")
print(f"   Vector size: {results['vector_size']}")
```

---
