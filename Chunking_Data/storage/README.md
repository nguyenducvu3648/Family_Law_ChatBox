# 💾 Storage - Data Persistence

> **Vai trò**: Lưu/đọc/merge JSON và kết nối Qdrant

---

## Storage layer xử lý mọi thứ liên quan đến:

```
✅ Lưu chunks ra JSON
✅ Đọc chunks từ JSON
✅ Merge nhiều JSON files
✅ Kết nối Qdrant vector database
✅ Upload embeddings lên Qdrant
```

---

## 📁 Files Trong Storage

### 1. `json_handler.py` - Xử Lý JSON

**Chức năng**: Save/Load/Merge chunks JSON files

#### **Functions**:

### `save_chunks_to_json(chunks, file_path)`

```python
Lưu chunks ra JSON file

Args:
    chunks: List[Dict] - Danh sách chunks
    file_path: str - Đường dẫn output

Output:
    File JSON với format đẹp (indent=2, ensure_ascii=False)
```

**Example**:

```python
from Chunking_Data.storage.json_handler import save_chunks_to_json

chunks = [
    {'id': 'LAW-D1', 'content': '...', 'metadata': {...}},
    {'id': 'LAW-D2', 'content': '...', 'metadata': {...}},
]

save_chunks_to_json(chunks, "data/BDS/law_chunks.json")
```

---

### `load_chunks_from_json(file_path)`

```python
Đọc chunks từ JSON file

Args:
    file_path: str - Đường dẫn file

Returns:
    List[Dict] - Danh sách chunks

Raises:
    FileNotFoundError: File không tồn tại
```

**Example**:

```python
from Chunking_Data.storage.json_handler import load_chunks_from_json

chunks = load_chunks_from_json("data/BDS_merged.json")
print(f"Loaded {len(chunks)} chunks")
```

---

### `merge_chunk_files(file_paths, remove_duplicates=True, verbose=True)`

```python
Merge nhiều JSON files thành 1 list

Args:
    file_paths: List[str] - Danh sách paths
    remove_duplicates: bool - Loại bỏ duplicate by ID (default: True)
    verbose: bool - Log chi tiết

Returns:
    List[Dict] - Merged chunks
```

**Logic**:

```
For each file:
  1. Load chunks
  2. Check duplicate by chunk['id']
  3. Add to merged list nếu chưa có
  ↓
Return merged chunks
```

**Example**:

```python
from Chunking_Data.storage.json_handler import merge_chunk_files

files = [
    "data/BDS/law1_chunk.json",
    "data/BDS/law2_chunk.json",
    "data/BDS/law3_chunk.json"
]

merged = merge_chunk_files(files, remove_duplicates=True)
print(f"Merged {len(merged)} chunks from {len(files)} files")

# Lưu merged file
save_chunks_to_json(merged, "data/BDS_merged.json")
```

---

### `get_file_stats(file_path)`

```python
Lấy statistics từ chunk file

Returns:
    {
        'total_chunks': 393,
        'law_ids': ['LHNVDG', 'LXAYDUNG'],
        'categories': ['BDS'],
        'avg_content_length': 256.5,
        'file_size_mb': 2.3
    }
```

---

### 2. `qdrant_client.py` - Qdrant Operations

**Chức năng**: Kết nối và upload lên Qdrant vector database

#### **Functions**:

### `get_qdrant_client()`

```python
Kết nối Qdrant từ .env config

Requires:
    QDRANT_URL=https://xxx.qdrant.io
    QDRANT_API_KEY=xxx

Returns:
    QdrantClient instance
```

**Example**:

```python
from Chunking_Data.storage.qdrant_client import get_qdrant_client

client = get_qdrant_client()
# → Connected to Qdrant
```

---

### `ensure_collection(client, collection_name, vector_size)`

```python
Tạo collection (nếu chưa có)

Args:
    client: QdrantClient
    collection_name: str (VD: "paraphrase-vietnamese-law-BDS")
    vector_size: int (VD: 768, 1024)

Logic:
    If collection exists:
        → Skip
    Else:
        → Create với vector_size và distance: Cosine
```

---

### `ensure_or_append_collection(client, collection_name, vector_size, append_mode=True)`

```python
Tạo hoặc append vào collection

Args:
    append_mode: bool
        - True: Append vào collection đã có
        - False: Delete và tạo mới (force recreate)

Raises:
    ValueError: Vector size mismatch (collection cũ có dim khác)
```

**Example**:

```python
from Chunking_Data.storage.qdrant_client import (
    get_qdrant_client,
    ensure_or_append_collection
)

client = get_qdrant_client()

# Append mode (default)
ensure_or_append_collection(
    client,
    "paraphrase-vietnamese-law-BDS",
    vector_size=768,
    append_mode=True
)

# Force recreate (XÓA data cũ)
ensure_or_append_collection(
    client,
    "paraphrase-vietnamese-law-BDS",
    vector_size=768,
    append_mode=False  # → Delete & recreate
)
```

---

### `encode_texts(texts, model_info, device="cuda")`

```python
Encode texts thành embeddings

Args:
    texts: List[str] - Texts cần encode
    model_info: Dict - {'name': '...', 'type': 'transformers'}
    device: str - 'cuda' hoặc 'cpu'

Returns:
    np.ndarray - Shape (n_texts, embedding_dim)

Logic:
    1. Load model (transformers hoặc sentence_transformers)
    2. Encode từng batch (16 texts/batch)
    3. Clear GPU cache sau mỗi batch
```

**Example**:

```python
from Chunking_Data.storage.qdrant_client import encode_texts

texts = [
    "Điều 1. Nội dung điều 1",
    "Điều 2. Nội dung điều 2"
]

model_info = {
    'name': 'minhquan6203/paraphrase-vietnamese-law',
    'type': 'transformers'
}

embeddings = encode_texts(texts, model_info, device='cuda')
print(embeddings.shape)  # (2, 768)
```

---

### `upsert_embeddings_to_qdrant(client, collection_name, embeddings, law_docs, batch_size=100)`

```python
Upload embeddings + metadata lên Qdrant

Args:
    client: QdrantClient
    collection_name: str
    embeddings: np.ndarray - Vectors
    law_docs: List[Dict] - Chunks (với metadata)
    batch_size: int - Upload batch size (default: 100)

Logic:
    For each batch of 100 vectors:
        1. Create PointStruct với:
           - id: auto UUID
           - vector: embedding
           - payload: chunk metadata
        2. Upsert batch lên Qdrant
```

**Metadata được index**:

```python
payload = {
    'id': chunk['id'],
    'content': chunk['content'],
    'law_id': metadata['law_id'],
    'law_no': metadata['law_no'],
    'law_title': metadata['law_title'],
    'chapter': metadata['chapter'],
    'article_no': metadata['article_no'],
    'clause_no': metadata['clause_no'],
    'point_letter': metadata['point_letter'],
    'exact_citation': metadata['exact_citation'],
    ...
}
```

**Example**:

```python
from Chunking_Data.storage.qdrant_client import (
    get_qdrant_client,
    encode_texts,
    upsert_embeddings_to_qdrant
)

# 1. Load chunks
chunks = load_chunks_from_json("data/BDS.json")

# 2. Encode
texts = [c['content'] for c in chunks]
embeddings = encode_texts(texts, model_info, device='cuda')

# 3. Upload
client = get_qdrant_client()
upsert_embeddings_to_qdrant(
    client,
    collection_name="paraphrase-vietnamese-law-BDS",
    embeddings=embeddings,
    law_docs=chunks,
    batch_size=100
)
```

---

### `count_collection_points(client, collection_name)`

```python
Đếm số vectors trong collection

Returns:
    int - Số vectors
```

---

### `get_embedding_dimension(model_info)`

```python
Lấy embedding dimension của model

Returns:
    int - Vector size (VD: 768, 1024)
```

---
