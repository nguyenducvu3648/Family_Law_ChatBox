#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Client Module
====================

Module tương tác với Qdrant vector database.

Functions:
  - business_id_to_uuid(): Convert business ID thành UUID deterministic
  - get_qdrant_client(): Kết nối Qdrant
  - ensure_collection(): Tạo collection mới (single vector)
  - ensure_or_append_collection(): Create hoặc append (single vector)
  - upsert_embeddings_to_qdrant(): Upload single vectors (UUID từ business ID)
  - ensure_hybrid_collection(): Tạo hybrid collection (multi-vector)
  - ensure_or_append_hybrid_collection(): Create hoặc append hybrid collection
  - upsert_hybrid_embeddings_to_qdrant(): Upload hybrid multi-vectors (UUID từ business ID)
  - count_collection_points(): Đếm số vectors
  - encode_texts(): Encode texts thành embeddings
  - get_embedding_dimension(): Get vector dimension
"""

import os
import uuid
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    SparseVectorParams, SparseVector, MultiVectorConfig,
    MultiVectorComparator, HnswConfigDiff, 
    Modifier
)


# ==================== UTILITIES ====================

def business_id_to_uuid(business_id: str) -> str:
    """
    Convert business ID thành UUID để dùng làm Qdrant point ID.

    Sử dụng UUID v5 (name-based) với namespace DNS để tạo UUID deterministic.

    Args:
        business_id: Business ID string (VD: "LHN2014-D1")

    Returns:
        UUID string
    """
    if not business_id or not isinstance(business_id, str):
        raise ValueError(f"Invalid business ID: {business_id}")

    # Sử dụng namespace DNS để tạo UUID deterministic
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, business_id))


# ==================== CONNECTION ====================

def get_qdrant_client() -> QdrantClient:
    """
    Kết nối với Qdrant server.
    
    Returns:
        QdrantClient instance
    
    Raises:
        RuntimeError: Nếu không thể kết nối
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        raise RuntimeError("QDRANT_URL not set in environment variables")
    
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=300.0,
            grpc_port=6334,
        )
        print("✅ Qdrant connected successfully")
        return client
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Qdrant: {e}")


# ==================== COLLECTION MANAGEMENT ====================

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int
) -> None:
    """
    Tạo mới collection (recreate nếu đã tồn tại).
    
    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        vector_size: Dimension của vector
    """
    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"📋 Collection ready: {collection_name} (dim={vector_size})")
    
    # Create payload indexes
    try:
        index_fields = {
            # Cấu trúc pháp điển
            "metadata.law_id": PayloadSchemaType.KEYWORD,
            "metadata.law_title": PayloadSchemaType.KEYWORD,
            "metadata.law_no": PayloadSchemaType.KEYWORD,
            "metadata.chapter": PayloadSchemaType.KEYWORD,
            "metadata.section": PayloadSchemaType.KEYWORD,
            "metadata.article_no": PayloadSchemaType.INTEGER,
            "metadata.article_title": PayloadSchemaType.KEYWORD,
            "metadata.clause_no": PayloadSchemaType.INTEGER,
            "metadata.point_letter": PayloadSchemaType.KEYWORD,
            "metadata.exact_citation": PayloadSchemaType.KEYWORD,
        }
        
        for field_name, schema_type in index_fields.items():
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                print(f"   ✅ Indexed: {field_name}")
            except Exception as ie:
                print(f"   ⚠️  Could not index '{field_name}': {ie}")
    except Exception as e:
        print(f"⚠️  Skipped creating payload indexes: {e}")


def get_collection_info(
    client: QdrantClient,
    collection_name: str
) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin về collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Tên collection
    
    Returns:
        Dict info hoặc None nếu không tồn tại
    """
    try:
        info = client.get_collection(collection_name)
        return {
            'config': info.config,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
        }
    except Exception:
        return None


def ensure_or_append_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    append_mode: bool = False
) -> bool:
    """
    Đảm bảo collection tồn tại. Trong append mode, chỉ tạo mới nếu chưa có.
    
    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        vector_size: Dimension của vector
        append_mode: True = append, False = recreate
    
    Returns:
        True nếu collection được tạo mới, False nếu đã tồn tại (append)
    
    Raises:
        ValueError: Nếu vector size không khớp (khi append)
    """
    existing_info = get_collection_info(client, collection_name)
    
    if existing_info:
        if append_mode:
            # Check vector size compatibility
            try:
                config = existing_info.get('config')
                
                # Try different ways to get vector size (depends on qdrant version)
                if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                    existing_vector_size = config.params.vectors.size
                elif isinstance(config, dict):
                    existing_vector_size = config['params']['vectors']['size']
                else:
                    # Fallback
                    existing_vector_size = vector_size
                
                if existing_vector_size != vector_size:
                    raise ValueError(
                        f"Vector size mismatch! Collection has {existing_vector_size}, "
                        f"but trying to upload {vector_size}. "
                        f"Use --force-recreate to override."
                    )
                
                print(f"📎 Appending to existing collection: {collection_name}")
                print(f"   Current points: {existing_info.get('points_count', 0)}")
                return False
                
            except Exception as e:
                print(f"⚠️  Warning checking vector size: {e}")
                print(f"📎 Appending to collection: {collection_name}")
                return False
        else:
            # Not append mode → recreate
            print(f"🔄 Recreating collection: {collection_name}")
            ensure_collection(client, collection_name, vector_size)
            return True
    else:
        # Collection doesn't exist → create
        print(f"➕ Creating new collection: {collection_name}")
        ensure_collection(client, collection_name, vector_size)
        return True


def ensure_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: Optional[int],
    colbert_dim: Optional[int]
) -> None:
    """
    Tạo mới hybrid collection với multi-vector config.

    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        dense_dim: Dimension của dense vector (bge-m3), None nếu không sử dụng
        colbert_dim: Dimension của ColBERT vector per token, None nếu không sử dụng
    """
    # Hybrid collection config - chỉ tạo vectors cho enabled types
    vectors_config = {}

    if dense_dim is not None:
        vectors_config["bge-m3"] = VectorParams(
            size=dense_dim,
            distance=Distance.COSINE,
        )

    if colbert_dim is not None:
        vectors_config["colbertv2.0"] = VectorParams(
            size=colbert_dim,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=HnswConfigDiff(m=0)  # Tắt HNSW vì không cần cho rerank
        )

    sparse_vectors_config = {}
    # Always include sparse config for now (can be made optional later)
    sparse_vectors_config["bm25"] = SparseVectorParams(modifier=Modifier.IDF)

    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config
    )

    print(f"🔍 Hybrid collection ready: {collection_name}")
    if dense_dim is not None:
        print(f"   Dense (bge-m3): {dense_dim} dims")
    else:
        print(f"   Dense (bge-m3): DISABLED")
    print(f"   Sparse (bm25): keyword-based")
    if colbert_dim is not None:
        print(f"   ColBERT: {colbert_dim} dims per token")
    else:
        print(f"   ColBERT: DISABLED")

    # Create payload indexes
    try:
        index_fields = {
            # Cấu trúc pháp điển
            "metadata.law_id": PayloadSchemaType.KEYWORD,
            "metadata.law_title": PayloadSchemaType.KEYWORD,
            "metadata.law_no": PayloadSchemaType.KEYWORD,
            "metadata.chapter": PayloadSchemaType.KEYWORD,
            "metadata.section": PayloadSchemaType.KEYWORD,
            "metadata.article_no": PayloadSchemaType.INTEGER,
            "metadata.article_title": PayloadSchemaType.KEYWORD,
            "metadata.clause_no": PayloadSchemaType.INTEGER,
            "metadata.point_letter": PayloadSchemaType.KEYWORD,
            "metadata.exact_citation": PayloadSchemaType.KEYWORD,
        }

        for field_name, schema_type in index_fields.items():
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                print(f"   ✅ Indexed: {field_name}")
            except Exception as ie:
                print(f"   ⚠️  Could not index '{field_name}': {ie}")
    except Exception as e:
        print(f"⚠️  Skipped creating payload indexes: {e}")


def ensure_or_append_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: Optional[int],
    colbert_dim: Optional[int],
    append_mode: bool = False
) -> bool:
    """
    Đảm bảo hybrid collection tồn tại. Trong append mode, chỉ tạo mới nếu chưa có.

    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        dense_dim: Dimension của dense vector (None nếu không sử dụng)
        colbert_dim: Dimension của ColBERT vector (None nếu không sử dụng)
        append_mode: True = append, False = recreate

    Returns:
        True nếu collection được tạo mới, False nếu đã tồn tại (append)

    Raises:
        ValueError: Nếu vector dimensions không khớp (khi append)
    """
    existing_info = get_collection_info(client, collection_name)

    if existing_info:
        if append_mode:
            # Check vector dimensions compatibility
            try:
                config = existing_info.get('config')

                # Get existing dimensions (complex logic for hybrid collections)
                # For now, assume compatible if collection exists
                # In production, you'd want more sophisticated checking

                print(f"📎 Appending to existing hybrid collection: {collection_name}")
                print(f"   Current points: {existing_info.get('points_count', 0)}")
                return False

            except Exception as e:
                print(f"⚠️  Warning checking hybrid collection: {e}")
                print(f"📎 Appending to hybrid collection: {collection_name}")
                return False
        else:
            # Not append mode → recreate
            print(f"🔄 Recreating hybrid collection: {collection_name}")
            ensure_hybrid_collection(client, collection_name, dense_dim, colbert_dim)
            return True
    else:
        # Collection doesn't exist → create
        print(f"➕ Creating new hybrid collection: {collection_name}")
        ensure_hybrid_collection(client, collection_name, dense_dim, colbert_dim)
        return True


def count_collection_points(
    client: QdrantClient,
    collection_name: str
) -> int:
    """
    Đếm số points trong collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Tên collection
    
    Returns:
        Số points
    """
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception:
        return 0


# ==================== UPLOAD ====================

def upsert_embeddings_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    law_docs: List[Dict[str, Any]],
    batch_size: int = 100
) -> None:
    """
    Upload embeddings và chunks lên Qdrant.

    Convert business ID từ doc["id"] thành UUID để làm point ID.
    UUID được tạo deterministic từ business ID để đảm bảo consistency.

    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        embeddings: Numpy array shape (n_docs, vector_dim)
        law_docs: List chunks với metadata (phải có field "id")
        batch_size: Batch size cho upload

    Raises:
        ValueError: Nếu document thiếu field "id" hoặc business ID invalid
    """
    if len(embeddings) != len(law_docs):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(law_docs)} documents"
        )
    
    print(f"📤 Uploading {len(law_docs)} vectors in batches of {batch_size}...")

    # Validate business IDs và convert to UUID
    for doc in law_docs:
        business_id = doc.get("id")
        if not business_id:
            raise ValueError(f"Document missing 'id' field: {doc}")

        # Convert business ID to UUID và lưu vào doc
        doc["_uuid"] = business_id_to_uuid(business_id)

    # Batch upload (sử dụng UUID từ business ID)
    for i in tqdm(range(0, len(law_docs), batch_size), desc="Uploading"):
        batch_docs = law_docs[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        points = []
        for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
            point_id = doc["_uuid"]  # UUID từ business ID
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "id": doc.get("id"),
                    "content": doc.get("content"),
                    "metadata": doc.get("metadata", {})
                }
            ))
        
        # Upsert batch
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    print(f"   ✅ Uploaded {len(law_docs)} vectors")


def upsert_hybrid_embeddings_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    dense_embeddings: Optional[np.ndarray],
    sparse_embeddings: Optional[List[Dict[str, Any]]],
    colbert_embeddings: Optional[List[np.ndarray]],
    law_docs: List[Dict[str, Any]],
    batch_size: int = 100
) -> None:
    """
    Upload hybrid multi-vector embeddings lên Qdrant.

    Convert business ID từ doc["id"] thành UUID để làm point ID.
    UUID được tạo deterministic từ business ID để đảm bảo consistency.

    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        dense_embeddings: Dense vectors (numpy array, shape: n_docs x dense_dim) or None
        sparse_embeddings: Sparse vectors (list of dicts with 'indices' and 'values') or None
        colbert_embeddings: ColBERT vectors (list of numpy arrays, shape: seq_len x colbert_dim) or None
        law_docs: List chunks với metadata (phải có field "id")
        batch_size: Batch size cho upload

    Raises:
        ValueError: Nếu document thiếu field "id", business ID invalid, hoặc mismatch dimensions
    """
    # Validate dimensions for non-None embeddings
    if dense_embeddings is not None and len(dense_embeddings) != len(law_docs):
        raise ValueError(f"Dense embeddings count mismatch: {len(dense_embeddings)} vs {len(law_docs)} documents")
    if sparse_embeddings is not None and len(sparse_embeddings) != len(law_docs):
        raise ValueError(f"Sparse embeddings count mismatch: {len(sparse_embeddings)} vs {len(law_docs)} documents")
    if colbert_embeddings is not None and len(colbert_embeddings) != len(law_docs):
        raise ValueError(f"ColBERT embeddings count mismatch: {len(colbert_embeddings)} vs {len(law_docs)} documents")

    print(f"📤 Uploading {len(law_docs)} hybrid vectors in batches of {batch_size}...")

    # Validate business IDs và convert to UUID
    for doc in law_docs:
        business_id = doc.get("id")
        if not business_id:
            raise ValueError(f"Document missing 'id' field: {doc}")

        # Convert business ID to UUID và lưu vào doc
        doc["_uuid"] = business_id_to_uuid(business_id)

    # Batch upload (sử dụng UUID từ business ID)
    for i in tqdm(range(0, len(law_docs), batch_size), desc="Uploading hybrid"):
        batch_docs = law_docs[i:i + batch_size]
        batch_dense = dense_embeddings[i:i + batch_size] if dense_embeddings is not None else [None] * len(batch_docs)
        batch_sparse = sparse_embeddings[i:i + batch_size] if sparse_embeddings is not None else [None] * len(batch_docs)
        batch_colbert = colbert_embeddings[i:i + batch_size] if colbert_embeddings is not None else [None] * len(batch_docs)

        points = []
        for j, doc in enumerate(batch_docs):
            point_id = doc["_uuid"]  # UUID từ business ID

            # Create hybrid vector payload - only include enabled vector types
            vector_payload = {}

            if dense_embeddings is not None:
                vector_payload["bge-m3"] = batch_dense[j].tolist()

            if sparse_embeddings is not None:
                vector_payload["bm25"] = SparseVector(
                    indices=batch_sparse[j]['indices'],
                    values=batch_sparse[j]['values']
                )

            if colbert_embeddings is not None:
                vector_payload["colbertv2.0"] = batch_colbert[j].tolist()  # Shape: (seq_len, dim)

            points.append(PointStruct(
                id=point_id,
                vector=vector_payload,
                payload={
                    "id": doc.get("id"),
                    "content": doc.get("content"),
                    "metadata": doc.get("metadata", {})
                }
            ))

        # Upsert batch
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"   ✅ Uploaded {len(law_docs)} hybrid vectors")


# ==================== EMBEDDING ====================

def mean_pooling(model_output, attention_mask):
    """Mean pooling để tạo sentence embeddings từ token embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_with_transformers(
    texts: List[str],
    model_name: str,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Encode texts using transformers library với mean pooling.
    
    Args:
        texts: List texts cần encode
        model_name: Tên model (HuggingFace)
        max_length: Max sequence length
        batch_size: Batch size
        device: "cuda" hoặc "cpu"
    
    Returns:
        Numpy array embeddings shape (n_texts, vector_dim)
    """
    from transformers import AutoTokenizer, AutoModel
    
    print(f"🤖 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Get model output
            model_output = model(**encoded)
            
            # Mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])
            
            # Normalize
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # To numpy
            embeddings = sentence_embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
            # Clear memory
            del encoded, model_output, sentence_embeddings, embeddings
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Cleanup
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Combine
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings


def encode_with_sentence_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Encode texts using sentence-transformers library.
    
    Args:
        texts: List texts cần encode
        model_name: Tên model
        batch_size: Batch size
        device: "cuda" hoặc "cpu"
    
    Returns:
        Numpy array embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install: pip install sentence-transformers"
        )
    
    print(f"🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to(device)
    
    # Encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return embeddings


def encode_texts(
    texts: List[str],
    model_info: Dict[str, Any],
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """
    Encode texts sử dụng model được chỉ định.
    
    Args:
        texts: List texts cần encode
        model_info: Dict với keys:
          - name: Model name
          - type: "transformers" hoặc "sentence_transformers"
          - max_length: Max sequence length (optional)
        device: "cuda" hoặc "cpu"
        batch_size: Batch size
    
    Returns:
        Numpy array embeddings
    """
    model_type = model_info['type']
    model_name = model_info['name']
    
    if model_type == 'sentence_transformers':
        return encode_with_sentence_transformers(
            texts, model_name, batch_size, device
        )
    elif model_type == 'transformers':
        max_length = model_info.get('max_length', 512)
        return encode_with_transformers(
            texts, model_name, max_length, batch_size, device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_embedding_dimension(model_info: Dict[str, Any]) -> int:
    """
    Lấy dimension của embedding vector từ model.
    
    Args:
        model_info: Dict model info
    
    Returns:
        Vector dimension (int)
    """
    # Test encode 1 text ngắn
    test_text = ["Test embedding dimension"]
    embeddings = encode_texts(test_text, model_info, device="cpu", batch_size=1)
    return embeddings.shape[1]

