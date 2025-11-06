"""
Evaluation module for Vietnamese Legal Document Chunking

This module provides AI-powered quality assurance for chunking results.
"""

from .ai_reviewer import (
    build_review_payload,
    call_gemini_review,
    GEMINI_MODEL_NAME,
    GEMINI_PROMPT
)

__all__ = [
    'build_review_payload',
    'call_gemini_review',
    'GEMINI_MODEL_NAME',
    'GEMINI_PROMPT'
]

