#!/usr/bin/env python3

import sys
import os
sys.path.append('Chunking_Data')

from Chunking_Data.storage.json_handler import load_chunks_from_json
from Chunking_Data.pipeline.hybrid_embedding_pipeline import HybridEmbeddingPipeline

# Load chunks
print("Loading chunks...")
chunks = load_chunks_from_json("data/merged_chunk_125008_061125.json")
print(f"Loaded {len(chunks)} chunks")

# Create pipeline
print("Creating hybrid pipeline...")
pipeline = HybridEmbeddingPipeline(
    dense_model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=16,
    verbose=True,
    vector_config={'dense': True, 'sparse': True, 'colbert': False}
)

# Process and upload
print("Processing and uploading...")
results = pipeline.process_and_upload(
    chunks=chunks,
    category="BDS",
    append_mode=False,  # Force recreate
    collection_name="BAAI_BDS_HYBRID_V2"
)

print("SUCCESS!")
print(f"Collection: {results['collection_name']}")
print(f"Vectors: {results['total_vectors']}")
if 'dense_dimension' in results:
    print(f"Dense dim: {results['dense_dimension']}")
