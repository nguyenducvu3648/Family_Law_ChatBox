# 📦 Core - Thuật Toán Chunking

> **Vai trò**: Chứa các thuật toán cốt lõi để chunk văn bản luật (pure functions)

---

## 📁 Files Trong Core

### 1. `docx_reader.py` - Đọc File Word

**Chức năng**: Đọc file .doc/.docx và trả về text

**Main Function**:

```python
def read_docx(file_path: str) → str
```

**Ví dụ**:

```python
from Chunking_Data.core.docx_reader import read_docx

text = read_docx("luat_nha_o_2023.docx")
print(f"Đọc được {len(text)} ký tự")
```

**Input**: Đường dẫn file
**Output**: Raw text (string)

---

### 2. `law_chunker.py` - Thuật Toán Chunking

**Chức năng**: Chia văn bản luật thành chunks theo cấu trúc pháp điển

**Main Function**:

```python
def chunk_law_document(
    text: str,
    law_id: str,
    law_no: str,
    law_title: str,
    issued_date: str,
    effective_date: str,
    expiry_date: str = None,
    signer: str = "",
    verbose: bool = True
) → (chunks: List[Dict], stats: Dict)
```

**Thuật Toán 2-Pass Parsing**:

#### **Pass 1: Pre-scan**

```
Scan toàn bộ văn bản để biết:
- Có bao nhiêu Chương? (chapters_set)
- Có bao nhiêu Điều? (articles_set)
→ Dùng để validate khi parsing
```

#### **Pass 2: Strict Parsing**

```
Parse từng dòng theo state machine:

State 1: Tìm Chương
  ↓
State 2: Tìm Điều
  ↓
State 3: Tìm Khoản
  ↓
State 4: Tìm Điểm (nếu có)
  ↓
Flush chunk khi đóng Khoản/Điểm/Điều
```

**Cấu Trúc Pháp Điển**:

```
Chương I – TÊN CHƯƠNG
  │
  ├── Mục 1 – TÊN MỤC (optional)
  │   │
  │   ├── Điều 1. Tên điều
  │   │   ├── (Intro text - optional)
  │   │   ├── Khoản 1. Nội dung khoản
  │   │   └── Khoản 2. Intro khoản có điểm:
  │   │       ├── a) Điểm a
  │   │       ├── b) Điểm b
  │   │       └── c) Điểm c
```

**Regex Patterns**:

```python
ARTICLE_RE = r'^Điều\s+(\d+)'           # Điều 1, Điều 24
CHAPTER_RE = r'^Chương\s+([IVXLCDM]+)'  # Chương I, Chương III
SECTION_RE = r'^Mục\s+(\d+)'            # Mục 1
CLAUSE_RE  = r'^\s*(\d+)\.'             # 1., 2., 3.
POINT_RE   = r'^\s*([a-z])[\).]'        # a), b), c.
```

**Đặc Điểm**:

1. **Clause Intro Injection**:

   ```
   Khoản 2. Intro khoản bao gồm:
     a) Điểm a
     b) Điểm b

   → Intro "Intro khoản bao gồm" được tiêm vào MỖI điểm
   ```

2. **Article Intro Handling**:

   ```
   Điều 5. Tên điều
   Intro điều không có khoản.

   → Tạo 1 chunk riêng cho intro này
   ```

3. **Citation Generation**:
   ```
   Mỗi chunk có exact_citation:
   - "Điều 5"
   - "Điều 5 khoản 2"
   - "Điều 5 khoản 2 điểm a."
   ```

**Chunk Format**:

```json
{
  "id": "LHNVDG-D5-K2-a",
  "content": "Điều 5. Tên điều Khoản 2. Intro, điểm a.\nNội dung điểm a...",
  "metadata": {
    "law_id": "LHNVDG",
    "law_no": "52/2014/QH13",
    "law_title": "Luật Hôn nhân và Gia đình",
    "chapter": "Chương I – NHỮNG QUY ĐỊNH CHUNG",
    "chapter_number": 1,
    "article_no": 5,
    "article_title": "Tên điều",
    "clause_no": 2,
    "point_letter": "a",
    "exact_citation": "Điều 5 khoản 2 điểm a.",
    "point_id": "dieu_5_khoan_2_diem_a",
    "issued_date": "2014-06-19",
    "effective_date": "2015-01-01"
  }
}
```

**Stats Output**:

```python
{
  "chapters_seen": ["Chương I – ...", "Chương II – ..."],
  "articles": 133,        # Số điều
  "article_intro": 39,    # Số điều có intro
  "clauses": 274,         # Số khoản
  "points": 80,           # Số điểm
  "citations": [...],     # List exact citations
  "total_chunks": 393
}
```

**Ví dụ sử dụng**:

```python
from Chunking_Data.core.law_chunker import chunk_law_document

text = """
Chương I – QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Luật này quy định về...

Điều 2. Nguyên tắc
1. Nguyên tắc thứ nhất...
2. Nguyên tắc thứ hai bao gồm:
a) Điểm a
b) Điểm b
"""

chunks, stats = chunk_law_document(
    text=text,
    law_id="LTEST",
    law_no="01/2024/QH15",
    law_title="Luật Test",
    issued_date="2024-01-01",
    effective_date="2024-07-01"
)

print(f"Tạo được {len(chunks)} chunks")
print(f"Stats: {stats}")
```

---

### 3. `law_id_generator.py` - Tạo Law ID

**Chức năng**: Tự động tạo law_id từ tên file

**Main Function**:

```python
generate_law_id(file_name: str) → str
```

**Logic**:

1. **Mapping Dictionary** (ưu tiên):

   ```python
   LAW_MAPPINGS = {
       'hôn nhân và gia đình': 'LHNVDG',
       'xây dựng': 'LXAYDUNG',
       'nhà ở': 'LNHAO',
       'đất đai': 'LDATDAI',
       ...
   }
   ```

2. **Xử lý Luật Sửa Đổi**:

   ```
   "Luật sửa đổi Luật Đất đai.docx"
   → Tìm tên luật gốc: "đất đai"
   → Mapping: LDATDAI
   → Output: "LSĐBSLDATDAI"
   ```

3. **Fallback - Chữ Cái Đầu**:
   ```
   "Luật Bảo vệ Môi trường 2020.docx"
   → Lấy chữ cái đầu: B, V, M, T
   → Output: "LBVMT"
   ```

**Examples**:

```python
from Chunking_Data.core.law_id_generator import generate_law_id

# Case 1: Trong mapping
law_id = generate_law_id("Luật Hôn nhân và Gia đình 2014.docx")
print(law_id)  # "LHNVDG"

# Case 2: Luật sửa đổi
law_id = generate_law_id("Luật sửa đổi, bổ sung Luật Đất đai.docx")
print(law_id)  # "LSĐBSLDATDAI"

# Case 3: Fallback
law_id = generate_law_id("Luật Bảo vệ Môi trường.docx")
print(law_id)  # "LBVMT"
```
