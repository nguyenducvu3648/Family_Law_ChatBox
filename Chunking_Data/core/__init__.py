"""
Core modules for document processing
"""

from .docx_reader import read_docx
from .law_chunker import chunk_law_document, normalize_lines
from .law_id_generator import generate_law_id

__all__ = [
    'read_docx',
    'chunk_law_document',
    'normalize_lines',
    'generate_law_id'
]

