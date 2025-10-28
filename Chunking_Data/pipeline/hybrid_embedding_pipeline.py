#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Embedding Pipeline Module
=================================

Pipeline để tạo và upload multi-vector embeddings cho hybrid search.
Hỗ trợ 3 loại vector: dense, sparse, và late interaction (ColBERT).

Features:
  - Dense embeddings: BAAI/bge-m3 hoặc custom model
  - Sparse embeddings: BM25-style từ fastembed
  - Late interaction embeddings: ColBERT từ fastembed
  - Hybrid collection với multi-vector config
  - Upload lên Qdrant với prefetch + fusion support
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm


class HybridEmbeddingPipeline:
    """
    Pipeline để tạo multi-vector embeddings cho hybrid search.

    Hỗ trợ 3 loại embeddings:
    - Dense: Semantic understanding (BAAI/bge-m3)
    - Sparse: Keyword-based (BM25)
    - Late Interaction: Contextual reranking (ColBERT)

    Usage:
        pipeline = HybridEmbeddingPipeline(
            dense_model_name="BAAI/bge-m3",
            verbose=True
        )
        pipeline.process_and_upload(chunks, category="BDS")
    """

    def __init__(
        self,
        dense_model_name: str,
        sparse_model_name: str = "Qdrant/bm25",
        colbert_model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
        batch_size: int = 16,
        verbose: bool = True
    ):
        """
        Initialize hybrid embedding pipeline.

        Args:
            dense_model_name: Tên model dense embedding (BAAI/bge-m3, etc.)
            sparse_model_name: Tên model sparse embedding (default: Qdrant/bm25)
            colbert_model_name: Tên model ColBERT (default: colbert-ir/colbertv2.0)
            device: Device để chạy ("cuda" hoặc "cpu")
            batch_size: Batch size cho embedding
            verbose: In log chi tiết
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.colbert_model_name = colbert_model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # Lazy load models
        self._dense_model = None
        self._sparse_model = None
        self._colbert_model = None

        # Vector dimensions (lazy load)
        self._dense_dim = None
        self._colbert_dim = None

    def _get_dense_model_info(self) -> Dict[str, Any]:
        """Lấy model info cho dense embedding"""
        # Auto-detect model type
        model_type = 'transformers'  # Default
        if ('sentence-transformers' in self.dense_model_name or
            'all-MiniLM' in self.dense_model_name or
            'BAAI' in self.dense_model_name or
            'paraphrase-multilingual' in self.dense_model_name):
            model_type = 'sentence_transformers'

        return {
            'name': self.dense_model_name,
            'type': model_type,
            'max_length': 512
        }

    def _load_dense_model(self):
        """Load dense embedding model (sentence-transformers)"""
        if self._dense_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install: pip install sentence-transformers"
                )

            if self.verbose:
                print(f"🤖 Loading dense model: {self.dense_model_name}")

            self._dense_model = SentenceTransformer(self.dense_model_name)
            self._dense_model.to(self.device)

    def _load_sparse_model(self):
        """Load sparse embedding model (fastembed)"""
        if self._sparse_model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed not installed. "
                    "Install: pip install fastembed"
                )

            if self.verbose:
                print(f"🔍 Loading sparse model: {self.sparse_model_name}")

            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)

    def _load_colbert_model(self):
        """Load ColBERT late interaction model (fastembed)"""
        if self._colbert_model is None:
            try:
                from fastembed import LateInteractionTextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed not installed. "
                    "Install: pip install fastembed"
                )

            if self.verbose:
                print(f"🎯 Loading ColBERT model: {self.colbert_model_name}")

            self._colbert_model = LateInteractionTextEmbedding(model_name=self.colbert_model_name)

    def get_dense_dimension(self) -> int:
        """Lấy dimension của dense vector"""
        if self._dense_dim is None:
            self._load_dense_model()
            # Test encode
            test_text = ["Test dense embedding"]
            embedding = self._dense_model.encode(test_text, convert_to_numpy=True)
            self._dense_dim = embedding.shape[1]

            if self.verbose:
                print(f"📏 Dense vector dimension: {self._dense_dim}")

        return self._dense_dim

    def get_colbert_dimension(self) -> int:
        """Lấy dimension của ColBERT vector (per token)"""
        if self._colbert_dim is None:
            self._load_colbert_model()
            # Test encode
            test_text = ["Test ColBERT embedding"]
            embedding = list(self._colbert_model.embed(test_text))[0]
            self._colbert_dim = len(embedding[0])  # embedding shape: (seq_len, dim)

            if self.verbose:
                print(f"📏 ColBERT vector dimension: {self._colbert_dim}")

        return self._colbert_dim

    def extract_texts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract text content từ chunks"""
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                texts.append(content)

        if self.verbose:
            print(f"📝 Extracted {len(texts)} texts for hybrid embedding")

        return texts

    def encode_dense(self, texts: List[str]) -> np.ndarray:
        """Encode texts thành dense embeddings"""
        self._load_dense_model()

        if self.verbose:
            print(f"🧠 Encoding dense vectors for {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Dense encoding"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._dense_model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)

        final_embeddings = np.vstack(embeddings)

        if self.verbose:
            print(f"   ✅ Dense embeddings shape: {final_embeddings.shape}")

        return final_embeddings

    def encode_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode texts thành sparse embeddings (BM25)"""
        self._load_sparse_model()

        if self.verbose:
            print(f"🔍 Encoding sparse vectors for {len(texts)} texts...")

        sparse_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Sparse encoding"):
            batch = texts[i:i + self.batch_size]
            batch_sparse = list(self._sparse_model.embed(batch))

            for sparse_vec in batch_sparse:
                # Convert to Qdrant SparseVector format
                sparse_embeddings.append({
                    'indices': sparse_vec.indices.tolist(),
                    'values': sparse_vec.values.tolist()
                })

        if self.verbose:
            print(f"   ✅ Sparse embeddings count: {len(sparse_embeddings)}")

        return sparse_embeddings

    def encode_colbert(self, texts: List[str]) -> List[np.ndarray]:
        """Encode texts thành ColBERT late interaction embeddings"""
        self._load_colbert_model()

        if self.verbose:
            print(f"🎯 Encoding ColBERT vectors for {len(texts)} texts...")

        colbert_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="ColBERT encoding"):
            batch = texts[i:i + self.batch_size]
            batch_colbert = list(self._colbert_model.embed(batch))

            for colbert_vec in batch_colbert:
                # colbert_vec is already numpy array with shape (seq_len, dim)
                colbert_embeddings.append(colbert_vec)

        if self.verbose:
            print(f"   ✅ ColBERT embeddings count: {len(colbert_embeddings)}")

        return colbert_embeddings

    def create_collection_name(self, category: str) -> str:
        """Tạo collection name cho hybrid search"""
        # Format: hybrid-{category}
        return f"hybrid-{category}"

    def upload_to_hybrid_collection(
        self,
        chunks: List[Dict[str, Any]],
        dense_embeddings: np.ndarray,
        sparse_embeddings: List[Dict[str, Any]],
        colbert_embeddings: List[np.ndarray],
        collection_name: str,
        append_mode: bool = True
    ):
        """
        Upload multi-vector embeddings lên hybrid collection.

        Args:
            chunks: List chunks với metadata
            dense_embeddings: Dense vectors (numpy array)
            sparse_embeddings: Sparse vectors (list of dicts)
            colbert_embeddings: ColBERT vectors (list of numpy arrays)
            collection_name: Tên collection
            append_mode: True = append, False = recreate
        """
        from ..storage.qdrant_client import (
            get_qdrant_client,
            ensure_or_append_hybrid_collection,
            upsert_hybrid_embeddings_to_qdrant,
            count_collection_points
        )

        if self.verbose:
            print(f"🗄️  Connecting to Qdrant for hybrid upload...")

        client = get_qdrant_client()

        if self.verbose:
            print(f"📋 Hybrid collection: {collection_name}")

        # Ensure hybrid collection
        try:
            dense_dim = self.get_dense_dimension()
            colbert_dim = self.get_colbert_dimension()
            ensure_or_append_hybrid_collection(
                client=client,
                collection_name=collection_name,
                dense_dim=dense_dim,
                colbert_dim=colbert_dim,
                append_mode=append_mode
            )
        except ValueError as e:
            raise ValueError(f"Failed to create/append hybrid collection: {e}")

        # Upload hybrid embeddings
        if self.verbose:
            print(f"📤 Uploading {len(chunks)} hybrid vectors...")

        upsert_hybrid_embeddings_to_qdrant(
            client=client,
            collection_name=collection_name,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            colbert_embeddings=colbert_embeddings,
            law_docs=chunks,
            batch_size=100
        )

        # Verify
        final_count = count_collection_points(client, collection_name)

        if self.verbose:
            print(f"✅ Hybrid upload successful!")
            print(f"   Collection: {collection_name}")
            print(f"   Total vectors: {final_count}")

    def process_and_upload(
        self,
        chunks: List[Dict[str, Any]],
        category: str,
        append_mode: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end hybrid embedding pipeline.

        Args:
            chunks: List chunks đã được tạo
            category: Category name (để tạo collection name)
            append_mode: True = append, False = recreate

        Returns:
            Dict kết quả với hybrid info
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 HYBRID EMBEDDING & UPLOAD PIPELINE")
            print(f"{'='*80}")
            print(f"📦 Chunks: {len(chunks)}")
            print(f"🧠 Dense model: {self.dense_model_name}")
            print(f"🔍 Sparse model: {self.sparse_model_name}")
            print(f"🎯 ColBERT model: {self.colbert_model_name}")
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

        # 2. Encode all embedding types
        dense_embeddings = self.encode_dense(texts)
        sparse_embeddings = self.encode_sparse(texts)
        colbert_embeddings = self.encode_colbert(texts)

        # 3. Create collection name
        collection_name = self.create_collection_name(category)

        # 4. Upload to hybrid collection
        self.upload_to_hybrid_collection(
            chunks=chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            colbert_embeddings=colbert_embeddings,
            collection_name=collection_name,
            append_mode=append_mode
        )

        # 5. Return results
        from ..storage.qdrant_client import get_qdrant_client, count_collection_points
        client = get_qdrant_client()
        final_count = count_collection_points(client, collection_name)

        results = {
            'collection_name': collection_name,
            'total_vectors': final_count,
            'dense_dimension': self.get_dense_dimension(),
            'colbert_dimension': self.get_colbert_dimension(),
            'dense_model': self.dense_model_name,
            'sparse_model': self.sparse_model_name,
            'colbert_model': self.colbert_model_name,
            'category': category,
            'embedding_types': ['dense', 'sparse', 'colbert']
        }

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎉 HYBRID PIPELINE COMPLETED!")
            print(f"{'='*80}")
            print(f"✅ Collection: {collection_name}")
            print(f"✅ Vectors: {final_count}")
            print(f"✅ Dense dim: {results['dense_dimension']}")
            print(f"✅ ColBERT dim: {results['colbert_dimension']}")
            print(f"🔍 Embedding types: {', '.join(results['embedding_types'])}")

        return results
