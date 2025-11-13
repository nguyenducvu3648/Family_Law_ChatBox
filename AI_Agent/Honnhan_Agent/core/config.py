import os
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

INTENT_DEBUG = os.getenv("INTENT_DEBUG", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CASUAL_MAX_WORDS = int(os.getenv("CASUAL_MAX_WORDS", "0").strip() or 0)
INTENT_RAW_PREVIEW_LIMIT = int(os.getenv("INTENT_RAW_PREVIEW_LIMIT", "240").strip() or 240)
INTENT_FALLBACK_CASUAL = os.getenv(
    "INTENT_FALLBACK_CASUAL",
    "Chào bạn, mình có thể hỗ trợ câu hỏi về Luật Hôn nhân & Gia đình. Bạn muốn hỏi nội dung gì?",
).strip()

if not (QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong tệp .env")
