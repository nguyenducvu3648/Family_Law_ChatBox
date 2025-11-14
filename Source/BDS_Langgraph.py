import os
import re
import json
import time
import logging
from datetime import datetime
from typing import TypedDict, Literal, Optional
from contextlib import contextmanager
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import gradio as gr

# LangChain & LangGraph
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Vector Store & Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import models

load_dotenv()

# ========================
# CONFIGURATION
# ========================
class Config:
    """Centralized configuration with validation"""
    
    @classmethod
    def load(cls):
        config = cls()
        
        # Required environment variables
        config.QDRANT_URL = os.getenv("QDRANT_URL")
        config.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        config.GEMINI_API_KEY1 = os.getenv("GEMINI_API_KEY1", os.getenv("GEMINI_API_KEY"))
        config.GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2", os.getenv("GEMINI_API_KEY"))
        
        if not all([config.QDRANT_URL, config.QDRANT_API_KEY, 
                    config.GEMINI_API_KEY1, config.GEMINI_API_KEY2]):
            raise ValueError("❌ Missing required environment variables!")
        
        # Optional with defaults
        config.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "BAAI_BDS_HYBRID")
        config.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        config.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        config.LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
        config.MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
        
        return config

try:
    config = Config.load()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
    print("Please set required environment variables:")
    print("  - QDRANT_URL")
    print("  - QDRANT_API_KEY")
    print("  - GEMINI_API_KEY1 (or GEMINI_API_KEY)")
    print("  - GEMINI_API_KEY2 (or GEMINI_API_KEY)")
    exit(1)

# ========================
# LOGGING WITH ROTATION
# ========================
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(LOGS_DIR, "bds_chatbot.log")

file_handler = RotatingFileHandler(
    LOG_FILENAME,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("🚀 BDS CHATBOT STARTING")
logger.info(f"📝 Log file: {LOG_FILENAME}")
logger.info("="*80)

# ========================
# THREAD-SAFE MODEL MANAGER
# ========================
class ModelManager:
    """Thread-safe singleton for model management"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._models_lock = threading.RLock()
        
        self.qdrant_client = None
        self.dense_model = None
        self.sparse_model = None
        self.late_model = None
        self.llm_route = None
        self.llm_answer = None
        self.executor = None
        self.compiled_graph = None
    
    def init_models(self):
        """Thread-safe model initialization"""
        with self._models_lock:
            if self.qdrant_client is None:
                logger.info("🔄 Initializing Qdrant client...")
                start = time.time()
                self.qdrant_client = QdrantClient(
                    url=config.QDRANT_URL,
                    api_key=config.QDRANT_API_KEY,
                    timeout=30
                )
                logger.info(f"✅ Qdrant initialized in {time.time()-start:.2f}s")
            
            if self.dense_model is None:
                logger.info("🔄 Loading dense model (BAAI/bge-m3)...")
                start = time.time()
                self.dense_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
                logger.info(f"✅ Dense model loaded in {time.time()-start:.2f}s")
            
            if self.sparse_model is None:
                logger.info("🔄 Loading sparse model (Qdrant/bm25)...")
                start = time.time()
                self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
                logger.info(f"✅ Sparse model loaded in {time.time()-start:.2f}s")
            
            if self.late_model is None:
                logger.info("🔄 Loading late interaction model (ColBERT)...")
                start = time.time()
                self.late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
                logger.info(f"✅ Late model loaded in {time.time()-start:.2f}s")
            
            if self.llm_route is None:
                logger.info("🔄 Initializing LLM for routing...")
                self.llm_route = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=config.GEMINI_API_KEY1,
                    temperature=0,
                    timeout=config.LLM_TIMEOUT,
                    max_retries=2
                )
                logger.info("✅ LLM route initialized")
            
            if self.llm_answer is None:
                logger.info("🔄 Initializing LLM for answer generation...")
                self.llm_answer = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=config.GEMINI_API_KEY2,
                    temperature=0.2,
                    timeout=config.LLM_TIMEOUT,
                    max_retries=2
                )
                logger.info("✅ LLM answer initialized")
            
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=config.MAX_WORKERS,
                    thread_name_prefix="rag_worker"
                )
                logger.info(f"✅ ThreadPoolExecutor created with {config.MAX_WORKERS} workers")
            
            if self.compiled_graph is None:
                logger.info("🔄 Compiling LangGraph workflow...")
                self.compiled_graph = build_graph()
                logger.info("✅ LangGraph compiled and cached")
    
    def shutdown(self):
        """Cleanup resources"""
        with self._models_lock:
            if self.executor:
                self.executor.shutdown(wait=True)
                logger.info("✅ ThreadPoolExecutor shutdown complete")
            
            if self.qdrant_client:
                self.qdrant_client.close()
                logger.info("✅ Qdrant client closed")

model_manager = ModelManager()

# ========================
# INPUT VALIDATION
# ========================
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_query(query: str) -> str:
    """Validate and sanitize user input"""
    if not query or not query.strip():
        raise ValidationError("Câu hỏi không được để trống")
    
    query = query.strip()
    
    if len(query) > config.MAX_QUERY_LENGTH:
        raise ValidationError(f"Câu hỏi quá dài (tối đa {config.MAX_QUERY_LENGTH} ký tự)")
    
    # Basic injection prevention
    dangerous_patterns = [
        r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
        r"(?i)(<script|javascript:)",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query):
            logger.warning(f"⚠️  Potentially malicious query detected: {query[:50]}")
            raise ValidationError("Câu hỏi chứa nội dung không hợp lệ")
    
    return query

# ========================
# STATE DEFINITION
# ========================
class GraphState(TypedDict):
    """State quản lý trong LangGraph"""
    original_query: str
    normalized_query: str
    optimized_query: str
    query_action: Literal["casual", "filter", "rag", "hybrid"]
    point_id: str | None
    law_title: str | None
    rag_docs: list[dict]
    filter_docs: list[dict]
    final_docs: list[dict]
    answer: str
    error: str | None
    debug_info: dict  # NEW: Debug information

# ========================
# TOOLS WITH RETRY
# ========================
def retry_on_failure(max_retries=2, delay=1):
    """Decorator for retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"⚠️  Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=2)
def filter_search(point_id: str = None, law_title: str = None, limit: int = 10) -> list[dict]:
    """Tool: Filter search with retry"""
    model_manager.init_models()
    
    start_time = time.time()
    logger.info(f"🔍 [FILTER] point_id={point_id}, law_title={law_title}")
    
    must_conditions = []
    
    if point_id:
        must_conditions.append(
            FieldCondition(key="metadata.point_id", match=MatchValue(value=point_id))
        )
    
    if law_title:
        must_conditions.append(
            FieldCondition(key="metadata.law_title", match=MatchValue(value=law_title))
        )
    
    if not must_conditions:
        logger.warning("⚠️  No filter conditions provided")
        return []
    
    filter_obj = Filter(must=must_conditions)
    
    scroll_result, _ = model_manager.qdrant_client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=limit,
        with_payload=True
    )
    
    docs = []
    for point in scroll_result:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        docs.append({
            "id": point.id,
            "point_id": meta.get("point_id", ""),
            "law_title": meta.get("law_title", ""),
            "chapter": meta.get("chapter", ""),
            "article_no": meta.get("article_no", ""),
            "article_title": meta.get("article_title", ""),
            "content": payload.get("content", ""),
            "score": 1.0,
            "source": "filter"
        })
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [FILTER] Found {len(docs)} docs in {elapsed:.3f}s")
    
    return docs

@retry_on_failure(max_retries=2)
def rag_search(query: str, top_k: int = 10) -> list[dict]:
    """Tool: Hybrid RAG search with retry"""
    model_manager.init_models()
    
    start_time = time.time()
    logger.info(f"🔍 [RAG] query='{query[:50]}...', top_k={top_k}")
    
    # Parallel embedding
    def embed_dense():
        return model_manager.dense_model.encode(query).tolist()
    
    def embed_sparse():
        return next(model_manager.sparse_model.query_embed(query))
    
    def embed_late():
        return next(model_manager.late_model.query_embed(query))
    
    future_dense = model_manager.executor.submit(embed_dense)
    future_sparse = model_manager.executor.submit(embed_sparse)
    future_late = model_manager.executor.submit(embed_late)
    
    dense_vec = future_dense.result()
    sparse_vec = future_sparse.result()
    late_vec = future_late.result()
    
    # Prefetch + ColBERT query
    prefetch = [
        models.Prefetch(query=dense_vec, using="bge-m3", limit=20),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist()
            ),
            using="bm25",
            limit=10
        )
    ]
    
    results = model_manager.qdrant_client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=prefetch,
        query=late_vec,
        using="colbertv2.0",
        with_payload=True,
        limit=top_k
    )
    
    docs = []
    for point in results.points:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        docs.append({
            "id": point.id,
            "point_id": meta.get("point_id", ""),
            "law_title": meta.get("law_title", ""),
            "chapter": meta.get("chapter", ""),
            "article_no": meta.get("article_no", ""),
            "article_title": meta.get("article_title", ""),
            "content": payload.get("content", ""),
            "score": point.score,
            "source": "rag"
        })
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [RAG] Found {len(docs)} docs in {elapsed:.3f}s")
    
    return docs

# ========================
# LANGGRAPH NODES
# ========================
def normalize_and_route_node(state: GraphState) -> GraphState:
    """Node 1+2: Normalize and route"""
    start_time = time.time()
    logger.info("📝 [NORMALIZE & ROUTE] Starting")
    
    original = state["original_query"]
    
    try:
        combined_prompt = ChatPromptTemplate.from_messages([
            ("system", """Phân tích câu hỏi pháp lý và trả về JSON.

ACTION:
- casual: Chào hỏi, cảm ơn
- filter: CÓ thông tin cụ thể (Điều X, Khoản Y)
- rag: Câu hỏi tổng quát
- hybrid: VỪA cụ thể VỪA cần tìm rộng

JSON OUTPUT:
{{"normalized_query": "...", "query_action": "...", "point_id": null, "law_title": null, "optimized_query": null}}"""),
            ("user", "{query}")
        ])
        
        parser = JsonOutputParser()
        chain = combined_prompt | model_manager.llm_route | parser
        
        result = chain.invoke({"query": original})
        
        elapsed = time.time() - start_time
        logger.info(f"✅ [NORMALIZE & ROUTE] {elapsed:.3f}s - action={result.get('query_action')}")
        
        return {
            **state,
            "normalized_query": result.get("normalized_query", original),
            "optimized_query": result.get("optimized_query", result.get("normalized_query", original)),
            "query_action": result.get("query_action", "rag"),
            "point_id": result.get("point_id"),
            "law_title": result.get("law_title"),
            "debug_info": {
                "normalize_time": elapsed,
                "action": result.get("query_action")
            }
        }
    
    except Exception as e:
        logger.error(f"❌ [NORMALIZE & ROUTE] Error: {e}", exc_info=True)
        return {
            **state,
            "normalized_query": original,
            "optimized_query": original,
            "query_action": "rag",
            "point_id": None,
            "law_title": None,
            "error": str(e),
            "debug_info": {"error": str(e)}
        }

def filter_node(state: GraphState) -> GraphState:
    """Node 3a: Filter search"""
    start_time = time.time()
    logger.info("🔎 [FILTER NODE] Starting")
    
    docs = filter_search(
        point_id=state.get("point_id"),
        law_title=state.get("law_title"),
        limit=10
    )
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [FILTER NODE] {elapsed:.3f}s - {len(docs)} docs")
    
    debug_info = state.get("debug_info", {})
    debug_info["filter_time"] = elapsed
    debug_info["filter_docs_count"] = len(docs)
    
    return {
        **state,
        "filter_docs": docs,
        "debug_info": debug_info
    }

def rag_node(state: GraphState) -> GraphState:
    """Node 3b: RAG search with dual query support"""
    start_time = time.time()
    logger.info("🔎 [RAG NODE] Starting")
    
    normalized_query = state["normalized_query"]
    optimized_query = state.get("optimized_query", normalized_query)
    action = state.get("query_action", "rag")
    
    # Dual query mode if optimized differs
    if action == "rag" and optimized_query and optimized_query != normalized_query:
        logger.info("⚡ [RAG NODE] Dual query mode")
        
        future1 = model_manager.executor.submit(rag_search, normalized_query, 10)
        future2 = model_manager.executor.submit(rag_search, optimized_query, 10)
        
        docs1 = future1.result()
        docs2 = future2.result()
        
        # Merge and deduplicate
        seen_ids = set()
        merged_docs = []
        
        for doc in docs2:
            doc_id = doc.get("id")
            if doc_id not in seen_ids:
                merged_docs.append(doc)
                seen_ids.add(doc_id)
        
        for doc in docs1:
            doc_id = doc.get("id")
            if doc_id not in seen_ids:
                merged_docs.append(doc)
                seen_ids.add(doc_id)
        
        docs = merged_docs[:10]
        logger.info(f"✅ [RAG NODE] Merged: {len(docs1)} + {len(docs2)} → {len(docs)} unique")
    else:
        logger.info("📝 [RAG NODE] Single query mode")
        docs = rag_search(normalized_query, top_k=10)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [RAG NODE] {elapsed:.3f}s - {len(docs)} docs")
    
    debug_info = state.get("debug_info", {})
    debug_info["rag_time"] = elapsed
    debug_info["rag_docs_count"] = len(docs)
    
    return {
        **state,
        "rag_docs": docs,
        "debug_info": debug_info
    }

def merge_docs_node(state: GraphState) -> GraphState:
    """Node 4: Merge documents"""
    start_time = time.time()
    logger.info("🔗 [MERGE] Starting")
    
    filter_docs = state.get("filter_docs", [])
    rag_docs = state.get("rag_docs", [])
    
    seen_ids = set()
    final_docs = []
    
    for doc in filter_docs:
        doc_id = doc.get("id")
        if doc_id not in seen_ids:
            final_docs.append(doc)
            seen_ids.add(doc_id)
    
    for doc in rag_docs:
        doc_id = doc.get("id")
        if doc_id not in seen_ids:
            final_docs.append(doc)
            seen_ids.add(doc_id)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [MERGE] {elapsed:.3f}s - {len(final_docs)} unique docs")
    
    debug_info = state.get("debug_info", {})
    debug_info["merge_time"] = elapsed
    debug_info["final_docs_count"] = len(final_docs)
    
    return {
        **state,
        "final_docs": final_docs,
        "debug_info": debug_info
    }

def answer_node(state: GraphState) -> GraphState:
    """Node 5: Generate answer"""
    start_time = time.time()
    logger.info("💬 [ANSWER] Starting")
    
    query = state["normalized_query"]
    docs = state.get("final_docs", [])
    
    if not docs:
        logger.warning("⚠️  [ANSWER] No documents found")
        return {
            **state,
            "answer": "⚠️ Xin lỗi, tôi không tìm thấy thông tin pháp lý phù hợp với câu hỏi của bạn.",
            "debug_info": {**state.get("debug_info", {}), "answer_time": 0}
        }
    
    # Build context
    context_parts = []
    for i, doc in enumerate(docs, 1):
        article = doc.get("article_no", "")
        title = doc.get("article_title", "")
        content = doc.get("content", "")
        citation = f"Điều {article}" if article else "N/A"
        context_parts.append(f"{i}. {citation} - {title}\n{content}\n")
    
    context = "\n".join(context_parts)
    
    # Generate answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia pháp lý BĐS Việt Nam. Trả lời dựa trên ngữ cảnh pháp luật được cung cấp.

YÊU CẦU:
- Trích dẫn chính xác Điều/Khoản
- Giải thích rõ ràng, dễ hiểu
- Không bịa đặt thông tin"""),
        ("user", "Câu hỏi: {query}\n\nNgữ cảnh:\n{context}\n\nTrả lời:")
    ])
    
    try:
        chain = answer_prompt | model_manager.llm_answer
        result = chain.invoke({"query": query, "context": context})
        answer = result.content.strip()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ [ANSWER] {elapsed:.3f}s - {len(answer)} chars")
        
        debug_info = state.get("debug_info", {})
        debug_info["answer_time"] = elapsed
        debug_info["answer_length"] = len(answer)
        
        return {
            **state,
            "answer": answer,
            "debug_info": debug_info
        }
    
    except Exception as e:
        logger.error(f"❌ [ANSWER] Error: {e}", exc_info=True)
        return {
            **state,
            "answer": f"❌ Lỗi sinh câu trả lời: {str(e)}",
            "error": str(e),
            "debug_info": {**state.get("debug_info", {}), "answer_error": str(e)}
        }

def casual_node(state: GraphState) -> GraphState:
    """Node: Casual conversation"""
    start_time = time.time()
    logger.info("💬 [CASUAL] Starting")
    
    query = state["normalized_query"]
    
    casual_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý pháp lý thân thiện. Trả lời ngắn gọn, lịch sự."),
        ("user", "{query}")
    ])
    
    chain = casual_prompt | model_manager.llm_answer
    result = chain.invoke({"query": query})
    
    elapsed = time.time() - start_time
    logger.info(f"✅ [CASUAL] {elapsed:.3f}s")
    
    return {
        **state,
        "answer": result.content.strip(),
        "final_docs": [],
        "debug_info": {**state.get("debug_info", {}), "casual_time": elapsed}
    }

# ========================
# ROUTING LOGIC
# ========================
def route_after_routing(state: GraphState) -> Literal["casual", "filter", "rag", "hybrid"]:
    action = state["query_action"]
    return action if action in ["casual", "filter", "rag", "hybrid"] else "rag"

# ========================
# BUILD GRAPH
# ========================
def build_graph():
    """Build LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("normalize_and_route", normalize_and_route_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("merge", merge_docs_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("casual", casual_node)
    
    # Define edges
    workflow.set_entry_point("normalize_and_route")
    
    workflow.add_conditional_edges(
        "normalize_and_route",
        route_after_routing,
        {
            "casual": "casual",
            "filter": "filter",
            "rag": "rag",
            "hybrid": "filter"
        }
    )
    
    workflow.add_edge("rag", "merge")
    workflow.add_edge("merge", "answer")
    workflow.add_edge("casual", END)
    workflow.add_edge("answer", END)
    
    workflow.add_conditional_edges(
        "filter",
        lambda state: "rag" if state["query_action"] == "hybrid" else "merge",
        {"rag": "rag", "merge": "merge"}
    )
    
    return workflow.compile()

# ========================
# MAIN EXECUTION
# ========================
def run_query(query: str):
    """Execute query with validation and metrics"""
    total_start = time.time()
    
    try:
        # Validate
        query = validate_query(query)
        
        # Initialize
        model_manager.init_models()
        
        # Execute
        graph = model_manager.compiled_graph
        
        initial_state = {
            "original_query": query,
            "normalized_query": "",
            "optimized_query": "",
            "query_action": "rag",
            "point_id": None,
            "law_title": None,
            "rag_docs": [],
            "filter_docs": [],
            "final_docs": [],
            "answer": "",
            "error": None,
            "debug_info": {}
        }
        
        logger.info(f"🚀 [QUERY START] {query[:100]}")
        
        result = graph.invoke(initial_state)
        
        elapsed = time.time() - total_start
        logger.info(f"✅ [QUERY END] {elapsed:.3f}s total")
        
        # Add total time to debug info
        result["debug_info"]["total_time"] = elapsed
        
        return result
    
    except ValidationError as e:
        logger.warning(f"⚠️  Validation error: {e}")
        return {
            "answer": f"❌ {str(e)}",
            "final_docs": [],
            "error": str(e),
            "debug_info": {"error": "validation", "message": str(e)}
        }
    
    except Exception as e:
        logger.error(f"❌ Execution error: {e}", exc_info=True)
        return {
            "answer": "❌ Xin lỗi, đã xảy ra lỗi hệ thống. Vui lòng thử lại sau.",
            "final_docs": [],
            "error": str(e),
            "debug_info": {"error": "execution", "message": str(e)}
        }

# ========================
# STREAMING ANSWER
# ========================
def generate_answer_stream(query: str, context: str):
    """Generator for streaming LLM response"""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia pháp lý BĐS Việt Nam. Trả lời dựa trên ngữ cảnh pháp luật được cung cấp.

YÊU CẦU:
- Trích dẫn chính xác Điều/Khoản
- Giải thích rõ ràng, dễ hiểu
- Không bịa đặt thông tin"""),
        ("user", "Câu hỏi: {query}\n\nNgữ cảnh:\n{context}\n\nTrả lời:")
    ])
    
    chain = answer_prompt | model_manager.llm_answer
    
    try:
        accumulated = ""
        for chunk in chain.stream({"query": query, "context": context}):
            if hasattr(chunk, 'content'):
                token = chunk.content
                accumulated += token
                yield accumulated
    except Exception as e:
        yield f"❌ Lỗi: {str(e)}"

# ========================
# GRADIO INTERFACE
# ========================
def format_docs_for_display(docs: list[dict]) -> str:
    """Format documents for display"""
    if not docs:
        return "📝 *Không có tài liệu tham khảo*"
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        law_title = doc.get("law_title", "N/A")
        article_no = doc.get("article_no", "N/A")
        article_title = doc.get("article_title", "N/A")
        chapter = doc.get("chapter", "")
        content = doc.get("content", "")
        score = doc.get("score", 0)
        source = doc.get("source", "unknown")
        
        # Color coding based on source
        source_emoji = "🔍" if source == "rag" else "🎯"
        
        formatted.append(f"""
### {source_emoji} {i}. {law_title} - Điều {article_no}

**📌 Tiêu đề:** {article_title}  
**📖 Chương:** {chapter}  
**🔢 Nguồn:** `{source.upper()}`  
**⭐ Độ liên quan:** {score:.4f}

**📄 Nội dung:**
> {content}

---
""")
    
    return "\n".join(formatted)

def format_debug_info(debug_info: dict) -> str:
    """Format debug information"""
    if not debug_info:
        return "⏱️ *Chưa có thông tin debug*"
    
    sections = []
    
    # Timing information
    if "total_time" in debug_info:
        sections.append(f"### ⏱️ Thời gian xử lý\n")
        sections.append(f"- **Tổng thời gian:** {debug_info['total_time']:.3f}s")
        
        if "normalize_time" in debug_info:
            sections.append(f"- **Phân tích câu hỏi:** {debug_info['normalize_time']:.3f}s")
        if "filter_time" in debug_info:
            sections.append(f"- **Tìm kiếm filter:** {debug_info['filter_time']:.3f}s")
        if "rag_time" in debug_info:
            sections.append(f"- **Tìm kiếm RAG:** {debug_info['rag_time']:.3f}s")
        if "merge_time" in debug_info:
            sections.append(f"- **Gộp tài liệu:** {debug_info['merge_time']:.3f}s")
        if "answer_time" in debug_info:
            sections.append(f"- **Sinh câu trả lời:** {debug_info['answer_time']:.3f}s")
        if "casual_time" in debug_info:
            sections.append(f"- **Xử lý casual:** {debug_info['casual_time']:.3f}s")
    
    # Action information
    if "action" in debug_info:
        sections.append(f"\n### 🎯 Loại xử lý\n")
        action_map = {
            "casual": "💬 Trò chuyện thông thường",
            "filter": "🎯 Tìm kiếm theo điều khoản cụ thể",
            "rag": "🔍 Tìm kiếm ngữ nghĩa",
            "hybrid": "⚡ Tìm kiếm kết hợp"
        }
        sections.append(f"- **Action:** {action_map.get(debug_info['action'], debug_info['action'])}")
    
    # Document counts
    if any(k in debug_info for k in ["filter_docs_count", "rag_docs_count", "final_docs_count"]):
        sections.append(f"\n### 📊 Số lượng tài liệu\n")
        if "filter_docs_count" in debug_info:
            sections.append(f"- **Filter:** {debug_info['filter_docs_count']} tài liệu")
        if "rag_docs_count" in debug_info:
            sections.append(f"- **RAG:** {debug_info['rag_docs_count']} tài liệu")
        if "final_docs_count" in debug_info:
            sections.append(f"- **Tổng (unique):** {debug_info['final_docs_count']} tài liệu")
    
    # Answer info
    if "answer_length" in debug_info:
        sections.append(f"\n### 📝 Thông tin câu trả lời\n")
        sections.append(f"- **Độ dài:** {debug_info['answer_length']} ký tự")
    
    # Error information
    if "error" in debug_info:
        sections.append(f"\n### ⚠️ Lỗi\n")
        sections.append(f"- **Loại:** {debug_info.get('error', 'unknown')}")
        if "message" in debug_info:
            sections.append(f"- **Chi tiết:** {debug_info['message']}")
    
    return "\n".join(sections)

def chatbot_interface_streaming(message: str, history: list):
    """Gradio streaming interface with enhanced UI"""
    if not message.strip():
        yield "⚠️ Vui lòng nhập câu hỏi của bạn.", "📝 *Không có tài liệu*", "⏱️ *Chưa có thông tin debug*"
        return
    
    try:
        logger.info(f"🌐 [GRADIO] New request: {message[:100]}")
        
        # Step 1: Initialize
        yield "🔄 **Đang khởi tạo hệ thống...**", "⏳ *Đang xử lý...*", "⏱️ *Đang khởi tạo...*"
        model_manager.init_models()
        
        # Step 2: Validate
        yield "✅ **Hệ thống đã sẵn sàng**\n\n🔍 **Đang phân tích câu hỏi...**", "⏳ *Đang xử lý...*", "⏱️ *Đang phân tích...*"
        
        initial_state = {
            "original_query": message,
            "normalized_query": "",
            "optimized_query": "",
            "query_action": "rag",
            "point_id": None,
            "law_title": None,
            "rag_docs": [],
            "filter_docs": [],
            "final_docs": [],
            "answer": "",
            "error": None,
            "debug_info": {}
        }
        
        # Step 3: Normalize and route
        state = normalize_and_route_node(initial_state)
        action = state.get("query_action", "rag")
        
        action_emoji = {
            "casual": "💬",
            "filter": "🎯",
            "rag": "🔍",
            "hybrid": "⚡"
        }
        
        action_name = {
            "casual": "Trò chuyện",
            "filter": "Tìm kiếm theo điều khoản",
            "rag": "Tìm kiếm ngữ nghĩa",
            "hybrid": "Tìm kiếm kết hợp"
        }
        
        debug_md = format_debug_info(state.get("debug_info", {}))
        
        if action == "casual":
            # Handle casual
            yield f"✅ **Phân tích hoàn tất**\n\n{action_emoji[action]} **Loại:** {action_name[action]}\n\n💬 **Đang trả lời...**", "📝 *Không có tài liệu (câu hỏi xã giao)*", debug_md
            
            casual_state = casual_node(state)
            yield casual_state["answer"], "📝 *Không có tài liệu tham khảo*", format_debug_info(casual_state.get("debug_info", {}))
            logger.info("✅ [GRADIO] Casual completed")
            return
        
        # Step 4: Retrieve documents
        yield f"✅ **Phân tích hoàn tất**\n\n{action_emoji[action]} **Loại:** {action_name[action]}\n\n🔍 **Đang tìm kiếm tài liệu pháp lý...**", "⏳ *Đang tìm kiếm...*", debug_md
        
        # Execute retrieval
        if action == "filter":
            state = filter_node(state)
            state = merge_docs_node(state)
        elif action == "rag":
            state = rag_node(state)
            state = merge_docs_node(state)
        elif action == "hybrid":
            state = filter_node(state)
            state = rag_node(state)
            state = merge_docs_node(state)
        
        docs = state.get("final_docs", [])
        formatted_docs = format_docs_for_display(docs)
        debug_md = format_debug_info(state.get("debug_info", {}))
        
        if not docs:
            yield "⚠️ Xin lỗi, tôi không tìm thấy thông tin pháp lý phù hợp với câu hỏi của bạn.", formatted_docs, debug_md
            logger.info("⚠️  [GRADIO] No docs found")
            return
        
        # Step 5: Generate answer with streaming
        yield f"✅ **Tìm thấy {len(docs)} tài liệu**\n\n✍️ **Đang soạn câu trả lời...**", formatted_docs, debug_md
        
        # Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            article = doc.get("article_no", "")
            title = doc.get("article_title", "")
            content = doc.get("content", "")
            citation = f"Điều {article}" if article else "N/A"
            context_parts.append(f"{i}. {citation} - {title}\n{content}\n")
        
        context = "\n".join(context_parts)
        query = state["normalized_query"]
        
        # Stream answer
        for partial_answer in generate_answer_stream(query, context):
            yield partial_answer, formatted_docs, debug_md
        
        # Final debug info update
        final_debug = state.get("debug_info", {})
        final_debug["answer_length"] = len(partial_answer) if 'partial_answer' in locals() else 0
        debug_md = format_debug_info(final_debug)
        
        yield partial_answer, formatted_docs, debug_md
        
        logger.info("✅ [GRADIO] Streaming completed")
        
    except ValidationError as e:
        error_msg = f"❌ **Lỗi xác thực:** {str(e)}"
        logger.warning(f"⚠️  [GRADIO] Validation error: {e}")
        yield error_msg, "📝 *Không có tài liệu*", format_debug_info({"error": "validation", "message": str(e)})
    
    except Exception as e:
        error_msg = f"❌ **Lỗi hệ thống:** Đã xảy ra lỗi. Vui lòng thử lại sau."
        logger.error(f"❌ [GRADIO] Error: {e}", exc_info=True)
        yield error_msg, "📝 *Không có tài liệu*", format_debug_info({"error": "execution", "message": str(e)})

def create_gradio_app():
    """Create enhanced Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .header-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background: #f8f9fa;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(
        title="🏢 Chatbot Pháp Luật BĐS",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=custom_css
    ) as app:
        
        # Header
        gr.Markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 class="header-text">🏢 Chatbot Pháp Luật Bất Động Sản Việt Nam</h1>
            <p style="font-size: 1.1em; color: #64748b;">
                Hệ thống tư vấn pháp luật thông minh với <strong>Hybrid RAG</strong> và <strong>LangGraph</strong>
            </p>
        </div>
        """)
        
        # Status indicator
        with gr.Row():
            with gr.Column(scale=1):
                status_box = gr.Markdown("""
                <div class="status-box">
                    <h3>📊 Trạng thái hệ thống</h3>
                    <ul>
                        <li>✅ <strong>Qdrant:</strong> Sẵn sàng</li>
                        <li>✅ <strong>LLM:</strong> Gemini 2.5 Flash</li>
                        <li>✅ <strong>Embeddings:</strong> BGE-M3 + BM25 + ColBERT</li>
                        <li>⚡ <strong>Streaming:</strong> Bật</li>
                    </ul>
                </div>
                """)
        
        gr.Markdown("---")
        
        # Main interface
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Đặt câu hỏi của bạn")
                
                question_input = gr.Textbox(
                    label="",
                    placeholder="Ví dụ: Điều 41 Khoản 2 Điểm c Luật Đất đai quy định gì?",
                    lines=4,
                    show_label=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Gửi câu hỏi", variant="primary", size="lg")
                    clear_btn = gr.Button("🔄 Xóa", variant="secondary", size="lg")
                
                gr.Markdown("### 📝 Ví dụ câu hỏi")
                
                example_buttons = []
                examples = [
                    "Điều 41 Khoản 2 Điểm c Luật Đất đai quy định gì?",
                    "Quy trình chuyển nhượng đất là gì?",
                    "Điều kiện để được cấp giấy chứng nhận quyền sử dụng đất?",
                    "Thời hạn sử dụng đất ở là bao lâu?",
                    "Ai được phép chuyển nhượng quyền sử dụng đất?"
                ]
                
                for example in examples:
                    btn = gr.Button(f"💡 {example[:60]}...", size="sm")
                    example_buttons.append((btn, example))
            
            # Right column - Debug info
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Thông tin Debug")
                debug_output = gr.Markdown(
                    value="⏱️ *Chưa có thông tin debug*",
                    label="Debug Info"
                )
        
        gr.Markdown("---")
        
        # Answer section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✅ Câu trả lời (⚡ Real-time Streaming)")
                answer_output = gr.Markdown(
                    value="*Câu trả lời sẽ hiển thị ở đây với hiệu ứng streaming...*",
                    label="Câu trả lời"
                )
        
        # Documents section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📚 Tài liệu pháp lý tham khảo")
                docs_output = gr.Markdown(
                    value="*Các văn bản pháp luật liên quan sẽ hiển thị ở đây...*",
                    label="Tài liệu"
                )
        
        # Footer
        gr.Markdown("""
        ---
        ### 💡 Hướng dẫn sử dụng
        
        1. **Nhập câu hỏi** về pháp luật bất động sản vào ô trên
        2. **Nhấn "Gửi câu hỏi"** hoặc Enter để bắt đầu
        3. **Xem real-time streaming** câu trả lời và tài liệu tham khảo
        4. **Kiểm tra Debug Info** để xem chi tiết quá trình xử lý
        
        #### 🎯 Các loại câu hỏi được hỗ trợ:
        
        | Loại | Icon | Mô tả | Ví dụ |
        |------|------|-------|-------|
        | **Filter** | 🎯 | Tìm kiếm theo điều khoản cụ thể | "Điều 41 Khoản 2 Luật Đất đai" |
        | **RAG** | 🔍 | Tìm kiếm ngữ nghĩa tổng quát | "Quy trình chuyển nhượng đất" |
        | **Hybrid** | ⚡ | Kết hợp cả hai phương pháp | "Điều 10 áp dụng như thế nào?" |
        | **Casual** | 💬 | Trò chuyện thông thường | "Xin chào", "Cảm ơn" |
        
        #### 🔧 Công nghệ:
        - **LangGraph**: Quản lý workflow phức tạp
        - **Hybrid RAG**: Dense (BGE-M3) + Sparse (BM25) + Late Interaction (ColBERT)
        - **Gemini 2.5 Flash**: Sinh câu trả lời với streaming
        - **Qdrant**: Vector database
        
        #### 📊 Thông tin Debug:
        - ⏱️ Thời gian xử lý từng bước
        - 🎯 Loại xử lý được chọn
        - 📊 Số lượng tài liệu tìm thấy
        - 📝 Độ dài câu trả lời
        
        ---
        <div style="text-align: center; color: #64748b; padding: 20px;">
            <p>Phát triển bởi <strong>BDS Legal AI Team</strong> | Version 2.0 Production</p>
            <p>⚡ Powered by Gemini 2.5 Flash + LangGraph + Qdrant</p>
        </div>
        """)
        
        # Event handlers
        def set_example(example_text):
            return example_text
        
        for btn, example_text in example_buttons:
            btn.click(
                fn=lambda ex=example_text: ex,
                inputs=[],
                outputs=[question_input]
            )
        
        submit_btn.click(
            fn=chatbot_interface_streaming,
            inputs=[question_input, gr.State([])],
            outputs=[answer_output, docs_output, debug_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Câu trả lời sẽ hiển thị ở đây...*", "*Tài liệu sẽ hiển thị ở đây...*", "⏱️ *Chưa có thông tin debug*"),
            inputs=[],
            outputs=[question_input, answer_output, docs_output, debug_output]
        )
        
        question_input.submit(
            fn=chatbot_interface_streaming,
            inputs=[question_input, gr.State([])],
            outputs=[answer_output, docs_output, debug_output]
        )
    
    return app

# ========================
# GRACEFUL SHUTDOWN
# ========================
import atexit
import signal
import sys

shutdown_requested = False

def shutdown_handler(signum=None, frame=None):
    """Graceful shutdown"""
    global shutdown_requested
    logger.info("🛑 Shutting down gracefully...")
    shutdown_requested = True
    model_manager.shutdown()
    logger.info("✅ Shutdown complete")
    # Force exit after cleanup
    import os
    os._exit(0)

atexit.register(shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 BDS CHATBOT PRODUCTION v2.0 STARTING")
    logger.info("="*80)
    
    try:
        # Pre-initialize models
        logger.info("⚡ Pre-initializing models...")
        init_start = time.time()
        model_manager.init_models()
        init_time = time.time() - init_start
        logger.info(f"✅ Models ready in {init_time:.2f}s")
        logger.info("="*80)
        
        # Launch Gradio app
        logger.info("🌐 Launching Gradio interface...")
        logger.info(f"📝 Log file: {LOG_FILENAME}")
        
        app = create_gradio_app()
        app.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
        logger.info("🎉 Gradio app launched successfully!")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt received, shutting down...")
        shutdown_handler()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        shutdown_handler()