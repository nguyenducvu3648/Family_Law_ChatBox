#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload to Qdrant Script
========================

Script embed chunks và upload lên Qdrant vector database.

Usage:
    # Upload với default model
    python -m Chunking_Data.scripts.upload_qdrant --chunk-file data/BDS_merged.json --category BDS
    
    # Upload với model khác
    python -m Chunking_Data.scripts.upload_qdrant --chunk-file data/BDS.json --category BDS --model BAAI/bge-m3
    
    # Force recreate collection
    python -m Chunking_Data.scripts.upload_qdrant --chunk-file data/BDS.json --category BDS --force-recreate
"""

import argparse

from ..pipeline.embedding_pipeline import EmbeddingPipeline
from ..storage.json_handler import load_chunks_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks và upload lên Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Upload với default model
  python -m Chunking_Data.scripts.upload_qdrant --chunk-file data/BDS_merged.json --category BDS
  
  # Upload với model khác
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --model BAAI/bge-m3
  
  # Force recreate (XÓA data cũ)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --force-recreate
  
  # Custom collection name
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --collection-name my_custom_collection

  # Dry run (test mà không upload)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --dry-run

  # Dense + Sparse hybrid (keyword + semantic)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --dense-sparse

  # Dense + ColBERT hybrid (semantic + reranking)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --dense-colbert

  # Full hybrid (dense + sparse + colbert)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --hybrid

  # Only BM25 sparse retrieval
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --sparse-only
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--chunk-file",
        # required=True,
        default="data/luat_hon_nhan_gia_dinh_2014_chunk_180412_241025.json",
        
        help="Path to chunk JSON file"
    )
    
    parser.add_argument(
        "--category",
        # required=True,
        default="QDS",
        help="Category name for collection (e.g., BDS, QDS)"
    )

    parser.add_argument(
        "--collection-name",
        help="Custom collection name (overrides auto-generated name)"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Embedding model name"
    )

    # Vector type options (mutually exclusive)
    vector_group = parser.add_mutually_exclusive_group()
    vector_group.add_argument(
        "--dense-only",
        action="store_true",
        help="Chỉ tạo dense embeddings (semantic retrieval)"
    )
    vector_group.add_argument(
        "--sparse-only",
        action="store_true",
        help="Chỉ tạo sparse embeddings (BM25 keyword retrieval)"
    )
    vector_group.add_argument(
        "--colbert-only",
        action="store_true",
        help="Chỉ tạo ColBERT embeddings (contextual reranking)"
    )
    vector_group.add_argument(
        "--dense-sparse",
        action="store_true",
        help="Tạo dense + sparse embeddings (hybrid keyword + semantic)"
    )
    vector_group.add_argument(
        "--dense-colbert",
        action="store_true",
        help="Tạo dense + ColBERT embeddings (semantic + reranking)"
    )
    vector_group.add_argument(
        "--hybrid",
        action="store_true",
        help="Tạo tất cả 3 loại embeddings (dense + sparse + colbert) - DEFAULT"
    )
    
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for embedding (default: cuda)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16)"
    )
    
    # Upload options
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreate collection (XÓA DỮ LIỆU CŨ!)"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to existing collection (DEFAULT)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: chỉ encode, không upload lên Qdrant"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In log chi tiết"
    )
    
    args = parser.parse_args()

    # Sanitize collection name if provided
    if args.collection_name:
        # Replace invalid characters with underscores
        args.collection_name = args.collection_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        # Remove any other invalid characters, keep only alphanumeric, hyphens, underscores
        import re
        args.collection_name = re.sub(r'[^a-zA-Z0-9_-]', '', args.collection_name)

    # Determine vector types to create
    # Default to hybrid if no specific option selected
    vector_config = {
        'dense': False,
        'sparse': False,
        'colbert': False
    }

    if args.dense_only:
        vector_config['dense'] = True
    elif args.sparse_only:
        vector_config['sparse'] = True
    elif args.colbert_only:
        vector_config['colbert'] = True
    elif args.dense_sparse:
        vector_config['dense'] = True
        vector_config['sparse'] = True
    elif args.dense_colbert:
        vector_config['dense'] = True
        vector_config['colbert'] = True
    else:
        # Default: hybrid (all three) or dense only
        if args.hybrid:
            vector_config['dense'] = True
            vector_config['sparse'] = True
            vector_config['colbert'] = True
        else:
            # Default to dense only if no hybrid option
            vector_config['dense'] = True

    # Fix: --force-recreate override --append
    if args.force_recreate:
        args.append = False
    
    print("=" * 80)
    print("🚀 EMBED & UPLOAD TO QDRANT")
    print("=" * 80)
    print(f"📁 Chunk file: {args.chunk_file}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Category: {args.category}")
    if hasattr(args, 'collection_name') and args.collection_name:
        print(f"📋 Collection: {args.collection_name} (custom, sanitized)")
    print(f"⚙️  Device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")

    # Show vector types
    vector_types = []
    if vector_config['dense']:
        vector_types.append("dense")
    if vector_config['sparse']:
        vector_types.append("sparse (BM25)")
    if vector_config['colbert']:
        vector_types.append("colbert (rerank)")

    print(f"🔍 Vector types: {', '.join(vector_types)}")

    # Determine if using hybrid pipeline
    use_hybrid = vector_config['sparse'] or vector_config['colbert']
    if use_hybrid:
        print(f"🔄 Pipeline: HYBRID")
    else:
        print(f"🔄 Pipeline: STANDARD")
    
    if args.force_recreate:
        print(f"⚠️  FORCE RECREATE MODE - Will DELETE existing data!")
    elif args.append:
        print(f"📎 APPEND MODE (default) - Will append to existing collection")
    else:
        print(f"🔄 CREATE MODE - Will create new collection")
    
    if args.dry_run:
        print(f"🔍 DRY RUN MODE - No upload to Qdrant")
    
    print("=" * 80)
    
    try:
        # 1. Load chunks
        chunks = load_chunks_from_json(args.chunk_file)
        
        if not chunks:
            print("❌ No valid chunks found!")
            return
        
        if args.dry_run:
            print(f"🔍 DRY RUN: Would process {len(chunks)} chunks")
            print("   Skipping embedding and upload")
            return
        
        # 2. Create pipeline
        if use_hybrid:
            from ..pipeline.hybrid_embedding_pipeline import HybridEmbeddingPipeline
            pipeline = HybridEmbeddingPipeline(
                dense_model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                verbose=args.verbose,
                vector_config=vector_config
            )
        else:
            pipeline = EmbeddingPipeline(
                model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                verbose=args.verbose
            )
        
        # 3. Process and upload
        results = pipeline.process_and_upload(
            chunks=chunks,
            category=args.category,
            append_mode=args.append,
            collection_name=args.collection_name
        )
        
        # 4. Summary
        print(f"\n{'='*80}")
        print(f"🎉 SUCCESS!")
        print(f"{'='*80}")
        print(f"✅ Collection: {results['collection_name']}")
        print(f"✅ Total vectors: {results['total_vectors']}")

        if use_hybrid:
            # Hybrid pipeline results
            if 'dense_dimension' in results:
                print(f"✅ Dense dim: {results['dense_dimension']}")
            if 'colbert_dimension' in results:
                print(f"✅ ColBERT dim: {results['colbert_dimension']}")
            print(f"✅ Models: dense={results.get('dense_model', 'N/A')}, sparse={results.get('sparse_model', 'N/A')}, colbert={results.get('colbert_model', 'N/A')}")
        else:
            # Standard pipeline results
            if 'vector_dimension' in results:
                print(f"✅ Dimension: {results['vector_dimension']}")
                print(f"✅ Model: {results.get('model_name', 'N/A')}")
            else:
                print(f"✅ Results keys: {list(results.keys())}")  # Debug
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

