# 🚀 Quick Start Guide

## ⚡ Bắt đầu nhanh (5 phút)

### 1️⃣ Setup Environment

```bash
# Copy template và edit .env
cp Chunking_Data/env.example .env

# Edit .env với editor
# Set: QDRANT_URL và QDRANT_API_KEY
```

### 2️⃣ Validate Config

```bash
python -m Chunking_Data --validate-config
```

Nếu thấy ✅ thì OK!

### 3️⃣ Run Workflow (3 bước cơ bản)

```bash
# BƯỚC 1: Tìm files luật
python -m Chunking_Data.scripts.find_files

# BƯỚC 2: Chunk theo category
python -m Chunking_Data.scripts.chunk_documents --category BDS

# BƯỚC 3: Upload lên Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS/BDS_chunk_*.json \
    --category BDS
```

**Done!** 🎉 Data đã được upload lên Qdrant.

---

## 📋 Workflow khuyến nghị (4 bước với merge)

Workflow này tốt hơn khi có nhiều files:

```bash
# BƯỚC 1: Tìm files
python -m Chunking_Data.scripts.find_files

# BƯỚC 2: Chunk nhiều files
python -m Chunking_Data.scripts.chunk_documents --category BDS --verbose

# BƯỚC 3: Merge chunks (dễ quản lý hơn)
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS

# BƯỚC 4: Upload merged file
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS_merged_*.json \
    --category BDS
```

---

## 🎯 Common Tasks

### Chunk 1 file cụ thể

```bash
python -m Chunking_Data.scripts.chunk_documents \
    --file "path/to/law.docx" \
    --law-no "52/2014/QH13" \
    --issued-date "2014-06-19" \
    --effective-date "2015-01-01"
```

### Chunk với validation

```bash
python -m Chunking_Data.scripts.chunk_documents \
    --category BDS \
    --validate \
    --verbose
```

### Upload với model khác

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --model BAAI/bge-m3
```

### Force recreate collection

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --force-recreate
```

---

## 🆘 Quick Troubleshooting

### ❌ "QDRANT_URL not set"

**Fix**:

```bash
# Tạo .env file
echo "QDRANT_URL=https://your-url.com" > .env
echo "QDRANT_API_KEY=your-key" >> .env
```

### ❌ "Vector size mismatch"

**Fix**: Dùng `--force-recreate`

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --force-recreate
```

### ❌ "CUDA out of memory"

**Fix 1**: Giảm batch size

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --batch-size 8
```

**Fix 2**: Dùng CPU

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --device cpu
```

---

## 📚 Đọc thêm

- **README.md** - Complete guide
- **ARCHITECTURE.md** - Technical details
- **REFACTORING_SUMMARY.md** - What's changed

---

## 💡 Tips

✅ **Always run `find_files.py` first** để catalog files  
✅ **Use `--validate`** để check chunk quality  
✅ **Merge chunks** trước khi upload (easier to manage)  
✅ **Backup chunks JSON** files (quan trọng!)  
✅ **Use `--verbose`** để debug khi có vấn đề

---

**Need help?** Check README.md hoặc ARCHITECTURE.md
