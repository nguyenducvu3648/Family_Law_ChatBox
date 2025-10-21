#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Client Module
====================

Module tương tác với Qdrant vector database.

Functions:
  - get_qdrant_client(): Kết nối Qdrant
  - ensure_collection(): Tạo collection mới
  - ensure_or_append_collection(): Create hoặc append
  - upsert_embeddings_to_qdrant(): Upload vectors
  - count_collection_points(): Đếm số vectors
  - encode_texts(): Encode texts thành embeddings
  - get_embedding_dimension(): Get vector dimension
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType


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
    
    Args:
        client: QdrantClient instance
        collection_name: Tên collection
        embeddings: Numpy array shape (n_docs, vector_dim)
        law_docs: List chunks với metadata
        batch_size: Batch size cho upload
    """
    if len(embeddings) != len(law_docs):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(law_docs)} documents"
        )
    
    print(f"📤 Uploading {len(law_docs)} vectors in batches of {batch_size}...")
    
    # Get current max ID để tránh conflict
    try:
        existing_count = count_collection_points(client, collection_name)
        start_id = existing_count
    except Exception:
        start_id = 0
    
    # Batch upload
    for i in tqdm(range(0, len(law_docs), batch_size), desc="Uploading"):
        batch_docs = law_docs[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        points = []
        for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
            point_id = start_id + i + j
            
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

