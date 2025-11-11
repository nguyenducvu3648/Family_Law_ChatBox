#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Pipeline Module
==========================

Pipeline để embed chunks và upload lên Qdrant vector database.

Features:
  - Batch embedding với progress tracking
  - Upload to Qdrant với retry
  - Collection management (create/append/recreate)
  - Model dimension detection
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingPipeline:
    """
    Pipeline để embed chunks và upload lên Qdrant.
    
    Usage:
        pipeline = EmbeddingPipeline(
            model_name="minhquan6203/paraphrase-vietnamese-law",
            verbose=True
        )
        pipeline.process_and_upload(chunks, category="BDS")
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 16,
        verbose: bool = True
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            model_name: Tên model embedding (HuggingFace)
            device: Device để chạy ("cuda" hoặc "cpu")
            batch_size: Batch size cho embedding
            verbose: In log chi tiết
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Lazy load để tránh import khi không cần
        self._model_info = None
        self._vector_size = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Lấy model info (lazy load)"""
        if self._model_info is None:
            # Auto-detect model type
            model_type = 'transformers'  # Default
            if ('sentence-transformers' in self.model_name or 
                'all-MiniLM' in self.model_name or 
                'paraphrase-multilingual' in self.model_name):
                model_type = 'sentence_transformers'
            
            self._model_info = {
                'name': self.model_name,
                'type': model_type,
                'max_length': 512
            }
            
            if self.verbose:
                print(f"🤖 Model type: {model_type}")
        
        return self._model_info
    
    def get_vector_dimension(self) -> int:
        """
        Lấy dimension của embedding vector.
        
        Returns:
            Vector dimension (int)
        """
        if self._vector_size is None:
            from ..storage.qdrant_client import get_embedding_dimension
            model_info = self._get_model_info()
            self._vector_size = get_embedding_dimension(model_info)
            
            if self.verbose:
                print(f"📏 Vector dimension: {self._vector_size}")
        
        return self._vector_size
    
    def extract_texts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract text content từ chunks.
        
        Args:
            chunks: List chunks
        
        Returns:
            List texts để embed
        """
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                texts.append(content)
        
        if self.verbose:
            print(f"📝 Extracted {len(texts)} texts for embedding")
        
        return texts
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts thành embeddings.
        
        Args:
            texts: List texts cần encode
        
        Returns:
            Numpy array embeddings shape (n_texts, vector_dim)
        """
        from ..storage.qdrant_client import encode_texts as _encode_texts
        
        if self.verbose:
            print(f"🧠 Encoding {len(texts)} texts with {self.model_name}...")
        
        model_info = self._get_model_info()
        embeddings = _encode_texts(
            texts=texts,
            model_info=model_info,
            device=self.device,
            batch_size=self.batch_size
        )
        
        if self.verbose:
            print(f"   ✅ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def create_collection_name(self, category: str) -> str:
        """
        Tạo collection name từ model và category.
        
        Format: {model-name}-{category}
        
        Args:
            category: Category name (VD: "BDS", "QDS")
        
        Returns:
            Collection name string
        """
        # Clean model name
        clean_model = self.model_name.replace('/', '-').replace('_', '-')
        return f"{clean_model}-{category}"
    
    def upload_to_qdrant(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        collection_name: str,
        append_mode: bool = True
    ):
        """
        Upload embeddings và chunks lên Qdrant.
        
        Args:
            chunks: List chunks (với metadata)
            embeddings: Numpy array embeddings
            collection_name: Tên collection
            append_mode: True = append, False = recreate
        """
        from ..storage.qdrant_client import (
            get_qdrant_client,
            ensure_or_append_collection,
            upsert_embeddings_to_qdrant,
            count_collection_points
        )
        
        if self.verbose:
            print(f"🗄️  Connecting to Qdrant...")
        
        client = get_qdrant_client()
        
        if self.verbose:
            print(f"📋 Collection: {collection_name}")
        
        # Ensure collection
        try:
            vector_size = self.get_vector_dimension()
            ensure_or_append_collection(
                client, collection_name, vector_size, append_mode=append_mode
            )
        except ValueError as e:
            raise ValueError(f"Failed to create/append collection: {e}")
        
        # Upload
        if self.verbose:
            print(f"📤 Uploading {len(chunks)} vectors...")
        
        upsert_embeddings_to_qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            law_docs=chunks,
            batch_size=100  # Qdrant batch size
        )
        
        # Verify
        final_count = count_collection_points(client, collection_name)
        
        if self.verbose:
            print(f"✅ Upload successful!")
            print(f"   Collection: {collection_name}")
            print(f"   Total vectors: {final_count}")
    
    def process_and_upload(
        self,
        chunks: List[Dict[str, Any]],
        category: str,
        append_mode: bool = True,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end: Extract texts → Encode → Upload.
        
        Args:
            chunks: List chunks đã được tạo
            category: Category name (để tạo collection name nếu không có custom)
            append_mode: True = append, False = recreate
            collection_name: Custom collection name (optional)
        
        Returns:
            Dict kết quả với keys:
              - collection_name
              - total_vectors
              - vector_dimension
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 EMBEDDING & UPLOAD PIPELINE")
            print(f"{'='*80}")
            print(f"📦 Chunks: {len(chunks)}")
            print(f"🤖 Model: {self.model_name}")
            print(f"📂 Category: {category}")
            print(f"⚙️  Device: {self.device}")
            print(f"📊 Batch size: {self.batch_size}")
            
            if append_mode:
                print(f"📎 Mode: APPEND")
            else:
                print(f"🔄 Mode: RECREATE")
            print(f"{'='*80}")
        
        # 1. Extract texts
        texts = self.extract_texts(chunks)
        
        if not texts:
            raise ValueError("No valid texts to embed!")
        
        # 2. Get vector dimension
        vector_size = self.get_vector_dimension()
        
        # 3. Encode texts
        embeddings = self.encode_texts(texts)
        
        # 4. Create collection name
        if collection_name:
            # Use custom collection name
            final_collection_name = collection_name
        else:
            # Auto-generate collection name
            final_collection_name = self.create_collection_name(category)
        
        # 5. Upload to Qdrant
        self.upload_to_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=final_collection_name,
            append_mode=append_mode
        )

        # 6. Return results
        from ..storage.qdrant_client import get_qdrant_client, count_collection_points
        client = get_qdrant_client()
        final_count = count_collection_points(client, final_collection_name)

        results = {
            'collection_name': final_collection_name,
            'total_vectors': final_count,
            'vector_dimension': vector_size,
            'model_name': self.model_name,
            'category': category
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎉 PIPELINE COMPLETED!")
            print(f"{'='*80}")
            print(f"✅ Collection: {collection_name}")
            print(f"✅ Vectors: {final_count}")
            print(f"✅ Dimension: {vector_size}")
        
        return results

