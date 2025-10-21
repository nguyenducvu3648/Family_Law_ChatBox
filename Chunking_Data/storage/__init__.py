"""
Storage modules for data persistence
"""

from .json_handler import save_chunks_to_json, load_chunks_from_json, merge_chunk_files
from .qdrant_client import (
    get_qdrant_client,
    ensure_collection,
    ensure_or_append_collection,
    upsert_embeddings_to_qdrant,
    count_collection_points,
    get_embedding_dimension,
    encode_texts
)

__all__ = [
    'save_chunks_to_json',
    'load_chunks_from_json',
    'merge_chunk_files',
    'get_qdrant_client',
    'ensure_collection',
    'ensure_or_append_collection',
    'upsert_embeddings_to_qdrant',
    'count_collection_points',
    'get_embedding_dimension',
    'encode_texts'
]

