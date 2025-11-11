#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module
====================

Central configuration cho Chunking Data package.
"""

import os
from typing import Dict, Any

# ==================== DEFAULT PATHS ====================

DEFAULT_LAW_CONTENT_DIR = "law_content"
DEFAULT_DATA_FILES_DIR = "data_files"
DEFAULT_OUTPUT_DIR = "data"

# ==================== DEFAULT MODELS ====================

DEFAULT_EMBEDDING_MODEL = "minhquan6203/paraphrase-vietnamese-law"

# Các models được support
SUPPORTED_MODELS = {
    "minhquan6203/paraphrase-vietnamese-law": {
        "type": "transformers",
        "description": "Fine-tuned cho luật Việt Nam (recommended)",
        "dimension": 768
    },
    "BAAI/bge-m3": {
        "type": "transformers",
        "description": "Multilingual model (BAAI)",
        "dimension": 1024  # Model thực tế output 1024 dims
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "type": "sentence_transformers",
        "description": "Multilingual SentenceTransformer",
        "dimension": 768
    },
    "namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims": {
        "type": "transformers",
        "description": "Vietnamese Law 256-dim",
        "dimension": 256
    }
}

# ==================== DEVICE SETTINGS ====================

DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 16

# ==================== CATEGORY MAPPINGS ====================

CATEGORY_MAPPINGS = {
    "BDS": {
        "full_name": "Bất động sản",
        "folder_name": "BDS",
        "description": "Luật Bất động sản, Nhà ở, Đất đai"
    },
    "DN": {
        "full_name": "Doanh nghiệp",
        "folder_name": "DN",
        "description": "Luật Doanh nghiệp, Công ty"
    },
    "TM": {
        "full_name": "Thương mại",
        "folder_name": "TM",
        "description": "Luật Thương mại"
    },
    "QDS": {
        "full_name": "Quyền dân sự",
        "folder_name": "QDS",
        "description": "Luật Dân sự, Hôn nhân Gia đình"
    }
}

# ==================== QDRANT SETTINGS ====================

def get_qdrant_config() -> Dict[str, Any]:
    """
    Lấy Qdrant config từ environment variables.
    
    Returns:
        Dict với keys: url, api_key
    """
    return {
        "url": os.getenv("QDRANT_URL", ""),
        "api_key": os.getenv("QDRANT_API_KEY", ""),
        "timeout": 300.0,
        "grpc_port": 6334
    }

# ==================== HELPER FUNCTIONS ====================

def get_category_info(category_code: str) -> Dict[str, str]:
    """
    Lấy thông tin category.
    
    Args:
        category_code: Code category (VD: "BDS", "QDS")
    
    Returns:
        Dict info hoặc default info nếu không tìm thấy
    """
    return CATEGORY_MAPPINGS.get(category_code.upper(), {
        "full_name": category_code,
        "folder_name": category_code.upper(),
        "description": f"Category {category_code}"
    })


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Lấy thông tin model.
    
    Args:
        model_name: Tên model
    
    Returns:
        Dict info
    """
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    
    # Auto-detect cho model không có trong list
    model_type = 'transformers'
    if 'sentence-transformers' in model_name or 'all-MiniLM' in model_name:
        model_type = 'sentence_transformers'
    
    return {
        "type": model_type,
        "description": f"Model {model_name}",
        "dimension": None  # Will be detected automatically
    }


def validate_config() -> bool:
    """
    Validate config (check required environment variables).
    
    Returns:
        True nếu config hợp lệ, False nếu không
    """
    qdrant_config = get_qdrant_config()
    
    if not qdrant_config["url"]:
        print("⚠️  WARNING: QDRANT_URL not set in environment")
        print("   Set in .env file or export QDRANT_URL=<your_url>")
        return False
    
    return True


# ==================== DISPLAY ====================

def print_config_summary():
    """In ra config summary"""
    print("=" * 80)
    print("⚙️  CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"📁 Law Content Dir: {DEFAULT_LAW_CONTENT_DIR}")
    print(f"📂 Data Files Dir: {DEFAULT_DATA_FILES_DIR}")
    print(f"📦 Output Dir: {DEFAULT_OUTPUT_DIR}")
    print(f"🤖 Default Model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"⚙️  Default Device: {DEFAULT_DEVICE}")
    print(f"📊 Default Batch Size: {DEFAULT_BATCH_SIZE}")
    
    qdrant_config = get_qdrant_config()
    print(f"\n🗄️  Qdrant:")
    if qdrant_config["url"]:
        print(f"   URL: {qdrant_config['url']}")
        print(f"   API Key: {'Set' if qdrant_config['api_key'] else 'Not set'}")
    else:
        print(f"   ⚠️  Not configured")
    
    print(f"\n📂 Categories:")
    for code, info in CATEGORY_MAPPINGS.items():
        print(f"   {code}: {info['full_name']}")
    
    print(f"\n🤖 Supported Models: {len(SUPPORTED_MODELS)}")
    for model_name, info in SUPPORTED_MODELS.items():
        print(f"   - {model_name}")
        print(f"     Type: {info['type']}, Dim: {info['dimension']}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config_summary()
    print(f"\n✅ Config valid: {validate_config()}")

