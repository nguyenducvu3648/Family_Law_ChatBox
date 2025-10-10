from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL_ID, INTENT_SYSTEM_PROMPT

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
    system_instruction=INTENT_SYSTEM_PROMPT,
)
answer_model = genai.GenerativeModel(model_name=GEMINI_MODEL_ID)

rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rerank_model.eval()