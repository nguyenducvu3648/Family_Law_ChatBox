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
  
  # Dry run (test mà không upload)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --dry-run

  # Hybrid search mode (multi-vector)
  python -m Chunking_Data.scripts.upload_qdrant \\
      --chunk-file data/BDS.json \\
      --category BDS \\
      --hybrid \\
      --model BAAI/bge-m3
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
    
    # Model options
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Embedding model name"
    )

    # Hybrid search option
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search với multi-vector (dense + sparse + colbert)"
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
    
    # Fix: --force-recreate override --append
    if args.force_recreate:
        args.append = False
    
    print("=" * 80)
    print("🚀 EMBED & UPLOAD TO QDRANT")
    print("=" * 80)
    print(f"📁 Chunk file: {args.chunk_file}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Category: {args.category}")
    print(f"⚙️  Device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")

    if args.hybrid:
        print(f"🔍 Hybrid mode: ENABLED (multi-vector)")
    else:
        print(f"🔍 Hybrid mode: DISABLED (single vector)")
    
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
        if args.hybrid:
            from ..pipeline.hybrid_embedding_pipeline import HybridEmbeddingPipeline
            pipeline = HybridEmbeddingPipeline(
                dense_model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                verbose=args.verbose
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
            append_mode=args.append
        )
        
        # 4. Summary
        print(f"\n{'='*80}")
        print(f"🎉 SUCCESS!")
        print(f"{'='*80}")
        print(f"✅ Collection: {results['collection_name']}")
        print(f"✅ Total vectors: {results['total_vectors']}")

        if args.hybrid:
            print(f"✅ Dense dim: {results['dense_dimension']}")
            print(f"✅ ColBERT dim: {results['colbert_dimension']}")
            print(f"✅ Models: dense={results['dense_model']}, sparse={results['sparse_model']}, colbert={results['colbert_model']}")
        else:
            print(f"✅ Dimension: {results['vector_dimension']}")
            print(f"✅ Model: {results['model_name']}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

