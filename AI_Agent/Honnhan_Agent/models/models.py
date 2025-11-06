from qdrant_client import QdrantClient
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Thêm fastembed
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, GEMINI_API_KEY, GEMINI_MODEL_ID

# Bổ sung dòng này để load prompt intent
from core.prompt_loader import load_prompt

# Xác định thư mục gốc của agent
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# -> BASE_DIR = Family-law-chatbot/AI_Agent/Honnhan_Agent

# Gọi load_prompt với folder và loại prompt
INTENT_SYSTEM_PROMPT = load_prompt(BASE_DIR, "intent")

# Prompt cho sinh câu trả lời
ANSWER_PROMPT = load_prompt(BASE_DIR, "answer")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

# Load models từ fastembed (thay thế SentenceTransformer)
embedding_model_name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

dense_embedding_model = TextEmbedding(embedding_model_name)
sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
    system_instruction=INTENT_SYSTEM_PROMPT,
)
answer_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
    system_instruction=ANSWER_PROMPT,
    )

rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rerank_model.eval()