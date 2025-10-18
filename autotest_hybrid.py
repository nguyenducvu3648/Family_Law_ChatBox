import os
import json
import time
import argparse
import re
import numpy as np
from datetime import datetime
import asyncio

from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# py autotest_hybrid.py -i full.xlsx -o results_hybrid.json
# Load environment variables from .env file
load_dotenv()

# ================== CONFIG ==================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Load BAAI reranker (nên load global)
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rerank_model.eval()

# ================== INITIALIZATION ==================
def initialize_clients():
    """Initializes and returns Qdrant and SentenceTransformer clients."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file.")
    
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Create payload indexes if they don't exist
    try:
        # Index for article_no (integer)
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="article_no",
            field_type="integer"
        )
        print("Created index for article_no")
    except Exception as e:
        print(f"Index for article_no may already exist: {e}")
    
    try:
        # Index for clause_no (integer)
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="clause_no",
            field_type="integer"
        )
        print("Created index for clause_no")
    except Exception as e:
        print(f"Index for clause_no may already exist: {e}")
    
    try:
        # Index for point_letter (keyword)
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="point_letter",
            field_type="keyword"
        )
        print("Created index for point_letter")
    except Exception as e:
        print(f"Index for point_letter may already exist: {e}")
    
    try:
        # Index for chapter_number (integer)
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="chapter_number",
            field_type="integer"
        )
        print("Created index for chapter_number")
    except Exception as e:
        print(f"Index for chapter_number may already exist: {e}")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return qdrant_client, embedding_model

def embed_query(model, query):
    """Embeds a single query using the provided SentenceTransformer model."""
    return model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

def rerank_with_baai(query, docs, top_k=7): # Changed top_k to 7 as per request
    if not docs:
        return docs

    # Filter out docs with empty content
    docs = [d for d in docs if d.get("content", "").strip()]

    pairs = [(query, d["content"]) for d in docs]
    inputs = rerank_tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        scores = rerank_model(**inputs).logits.view(-1).float()

    # Gắn lại score vào docs
    for d, s in zip(docs, scores):
        d["baai_score"] = float(s)

    reranked = sorted(docs, key=lambda x: x["baai_score"], reverse=True)
    return reranked[:top_k]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

# Load all documents at startup for global BM25
def load_all_docs(client):
    docs = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            offset=offset
        )
        for point in points:
            p = point.payload or {}
            docs.append({
                "citation": p.get("exact_citation", ""),
                "chapter_number": p.get("chapter_number", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
            })
        offset = next_offset
        if offset is None:
            break
    return docs

# Global BM25 instance (will be initialized in main)
bm25_global = None
all_docs = []

# ================== LEGAL REFERENCE EXTRACTION AND COMPARISON ==================
LEGAL_BASE_PATTERN = re.compile(r"cơ\s*sở\s*pháp\s*lý\s*[:：]", re.IGNORECASE)

def extract_legal_refs(answer_text: str):
    """Extract legal references (Điều/Khoản/Điểm) from the 'Cơ sở pháp lý:' section.

    Returns a list of dicts like {"article": int, "clause": Optional[int], "point": Optional[str]}.
    Supports common formats and multiple refs separated by ';', ',', '\n', 'và'.
    """
    refs = []
    if not isinstance(answer_text, str) or not answer_text.strip():
        return refs

    # Focus only on the substring after 'Cơ sở pháp lý:' if present
    s = answer_text
    m_base = LEGAL_BASE_PATTERN.search(s)
    if m_base:
        s = s[m_base.end():]

    # Normalize spacing
    s_norm = re.sub(r"\s+", " ", s.lower()).strip()

    # Split on common delimiters to find individual reference chunks
    parts = re.split(r"[;\n\r]|\s+và\s+|\s*,\s*", s_norm)

    # Patterns to capture combinations in typical orders
    patt1 = re.compile(r"điểm\s*([a-z])\s*khoản\s*(\d+)\s*điều\s*(\d+)")
    patt2 = re.compile(r"khoản\s*(\d+)\s*điều\s*(\d+)\s*(?:điểm\s*([a-z]))?")
    patt3 = re.compile(r"điều\s*(\d+)(?:\s*khoản\s*(\d+))?(?:\s*điểm\s*([a-z]))?")

    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = patt1.search(p)
        if m:
            point, clause, article = m.group(1), m.group(2), m.group(3)
            return [{"article": int(article), "clause": int(clause), "point": point.lower()}]
        m = patt2.search(p)
        if m:
            clause, article, point = m.group(1), m.group(2), m.group(3)
            return [{
                "article": int(article),
                "clause": int(clause) if clause else None,
                "point": point.lower() if point else None,
            }]
        m = patt3.search(p)
        if m:
            article, clause, point = m.group(1), m.group(2), m.group(3)
            return [{
                "article": int(article),
                "clause": int(clause) if clause else None,
                "point": point.lower() if point else None,
            }]

    return refs

def _result_payload_triplet(result):
    # 'result' is already the document dictionary itself, not a Qdrant ScoredPoint
    # So, we can directly access its keys.
    art = result.get('article_no')
    cls = result.get('clause_no')
    pt  = result.get('point_letter')
    try:
        art = int(art) if art is not None and str(art).strip() != '' else None
    except Exception:
        art = None
    try:
        cls = int(cls) if cls is not None and str(cls).strip() != '' else None
    except Exception:
        cls = None
    pt = (str(pt).lower().strip() if pt is not None and str(pt).strip() != '' else None)
    return art, cls, pt

def _ref_matches_triplet(ref, triplet):
    """Return True if a retrieved (article, clause, point) matches the expected ref granularity."""
    ra, rc, rp = ref.get('article'), ref.get('clause'), ref.get('point')
    ta, tc, tp = triplet
    if not ta or not ra:
        return False
    if ta != ra:
        return False
    # If expected specifies clause, it must match
    if rc is not None and tc != rc:
        return False
    # If expected specifies point, it must match
    if rp is not None and tp != rp:
        return False
    return True

def check_refs_in_results(search_results, expected_refs):
    """Check if any expected legal reference is present in the search results.

    expected_refs: list of {article:int, clause:Optional[int], point:Optional[str]}.
    Returns (found: bool, score: Optional[float], matched_ref: Optional[dict], matched_doc: Optional[dict])
    """
    if not expected_refs:
        return False, None, None, None
    for res in search_results:
        trip = _result_payload_triplet(res)
        for ref in expected_refs:
            if _ref_matches_triplet(ref, trip):
                # Found a match; return with this result's score and basic doc info
                try:
                    score = float(getattr(res, 'score', None) or res.get('score'))
                except Exception:
                    score = None
                p = res.payload if hasattr(res, 'payload') else res.get('payload', {})
                doc = {
                    "article_no": p.get('article_no'),
                    "clause_no": p.get('clause_no'),
                    "point_letter": p.get('point_letter'),
                    "exact_citation": p.get('exact_citation'),
                }
                return True, score, ref, doc
    return False, None, None, None

# ================== HYBRID SEARCH (BM25 + EMBEDDING + RERANK) ==================
def _build_filter(query_text: str) -> Optional[Filter]:
    conds: List[FieldCondition] = []
    m = re.search(r"(?i)\bđiều\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="article_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bkhoản\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="clause_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bđiểm\s*([a-z])\b", query_text)
    if m:
        conds.append(FieldCondition(key="point_letter", match=MatchValue(value=m.group(1).lower())))
    m = re.search(r"(?i)\bchương\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="chapter_number", match=MatchValue(value=int(m.group(1)))))
    return Filter(must=conds) if conds else None

async def search_law_async(qdrant_client, embedding_model, query: str, top_k: int = 20, score_threshold: float = 0.3, case_type: str = "bm25_retrieval_rerank"):
    flt = _build_filter(query)
    has_filter = flt is not None and flt.must

    if case_type == "retrieval_only":
        # Only embedding retrieval
        vec = embed_query(embedding_model, query)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            with_payload=True,
            limit=top_k,
            query_filter=flt,
        )
        retrieved_docs = []
        for r in results.points:
            p = r.payload or {}
            retrieved_docs.append({
                "citation": p.get("exact_citation", ""),
                "chapter_number": p.get("chapter_number", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": float(r.score or 0.0),
            })
        return retrieved_docs

    elif case_type in ["bm25_retrieval", "bm25_retrieval_rerank"]:
        # Hybrid search logic
        async def bm25_search_task():
            if flt:
                scroll_res, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=flt,
                    limit=200,
                    with_payload=True,
                )
                filtered_docs = []
                for r in scroll_res:
                    p = r.payload or {}
                    filtered_docs.append({
                        "citation": p.get("exact_citation", ""),
                        "chapter_number": p.get("chapter_number", ""),
                        "article_no": p.get("article_no", ""),
                        "article_title": p.get("article_title", ""),
                        "clause_no": p.get("clause_no", ""),
                        "point_letter": p.get("point_letter", ""),
                        "content": (p.get("content") or "").strip(),
                    })
                tokenized_filtered = [tokenize(d['content']) for d in filtered_docs]
                bm25 = BM25Okapi(tokenized_filtered)
                docs_base = filtered_docs
            else:
                bm25 = bm25_global
                docs_base = all_docs

            tokenized_query = tokenize(query)
            bm25_scores = bm25.get_scores(tokenized_query)
            scored_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:50]
            bm25_docs = []
            for idx in scored_indices:
                if bm25_scores[idx] > 0:
                    d = docs_base[idx].copy()
                    d['bm25_score'] = float(bm25_scores[idx])
                    bm25_docs.append(d)
            return bm25_docs

        async def embedding_search_task():
            vec = embed_query(embedding_model, query)
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vec,
                with_payload=True,
                limit=30,
                query_filter=flt,
            )
            emb_docs = []
            for r in results.points:
                p = r.payload or {}
                emb_docs.append({
                    "citation": p.get("exact_citation", ""),
                    "chapter_number": p.get("chapter_number", ""),
                    "article_no": p.get("article_no", ""),
                    "article_title": p.get("article_title", ""),
                    "clause_no": p.get("clause_no", ""),
                    "point_letter": p.get("point_letter", ""),
                    "content": (p.get("content") or "").strip(),
                    "embedding_score": float(r.score or 0.0),
                })
            return emb_docs

        bm25_task = asyncio.create_task(bm25_search_task())
        embedding_task = asyncio.create_task(embedding_search_task())

        bm25_docs = await bm25_task
        emb_docs = await embedding_task

        all_unique = {}
        key_func = lambda d: (d.get('article_no', ''), d.get('clause_no', ''), d.get('point_letter', ''))
        for d in emb_docs:
            key = key_func(d)
            if key not in all_unique:
                all_unique[key] = d.copy()
            all_unique[key]['embedding_score'] = d['embedding_score']
            all_unique[key]['bm25_score'] = 0.0

        for d in bm25_docs:
            key = key_func(d)
            if key not in all_unique:
                all_unique[key] = d.copy()
            all_unique[key]['bm25_score'] = d['bm25_score']
            all_unique[key]['embedding_score'] = all_unique[key].get('embedding_score', 0.0)

        merged_docs = list(all_unique.values())

        if merged_docs:
            emb_scores = [d['embedding_score'] for d in merged_docs]
            bm25_scores = [d['bm25_score'] for d in merged_docs]
            min_emb, max_emb = min(emb_scores), max(emb_scores) if emb_scores else (0, 0)
            min_bm25, max_bm25 = min(bm25_scores), max(bm25_scores) if bm25_scores else (0, 0)

            for d in merged_docs:
                if max_emb > min_emb:
                    d['norm_emb'] = (d['embedding_score'] - min_emb) / (max_emb - min_emb)
                else:
                    d['norm_emb'] = 0.5 if d['embedding_score'] > 0 else 0.0
                if max_bm25 > min_bm25:
                    d['norm_bm25'] = (d['bm25_score'] - min_bm25) / (max_bm25 - min_bm25)
                else:
                    d['norm_bm25'] = 0.5 if d['bm25_score'] > 0 else 0.0

            if has_filter:
                alpha = 0.7
                beta = 0.3
            else:
                alpha = 0.4
                beta = 0.6

            for d in merged_docs:
                d['score'] = alpha * d['norm_emb'] + beta * d['norm_bm25']

            ranked = sorted(merged_docs, key=lambda d: d['score'], reverse=True)

            for d in ranked:
                d.pop('norm_emb', None)
                d.pop('norm_bm25', None)
                d.pop('embedding_score', None)
                d.pop('bm25_score', None)

            if case_type == "bm25_retrieval":
                return ranked[:top_k]
            else:  # bm25_retrieval_rerank
                selected = [d for d in ranked if d['score'] >= score_threshold][:20]
                if selected:
                    selected = rerank_with_baai(query, selected, top_k=top_k)
                return selected
        else:
            return []

def search_law_sync(qdrant_client, embedding_model, query: str, top_k: int = 15, score_threshold: float = 0.42, case_type: str = "bm25_retrieval_rerank"):
    """Synchronous wrapper for search_law_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(search_law_async(qdrant_client, embedding_model, query, top_k, score_threshold, case_type))


# ================== TEST EXECUTION ==================
def run_test(qdrant_client, embedding_model, test_data):
    """Runs the automated test for each query in the test data."""
    results = []
    
    cases = [
        ("retrieval_only", "retrieval_only"),
        ("bm25_retrieval", "bm25_retrieval"),
        ("bm25_retrieval_rerank", "bm25_retrieval_rerank")
    ]
    
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing queries"):
        query = row['query']
        answer_text = row.get('answer')
        expected_refs = extract_legal_refs(answer_text) if isinstance(answer_text, str) else []
        
        for case_name, case_type in cases:
            if case_name == "retrieval_only":
                top_k = 15
            elif case_name == "bm25_retrieval":
                top_k = 20
            elif case_name == "bm25_retrieval_rerank":
                top_k = 10
            else:
                top_k = 7
            start_time = time.time()
            # Call the hybrid search function
            retrieved_docs = search_law_sync(qdrant_client, embedding_model, query, top_k=top_k, case_type=case_type)
            search_time = time.time() - start_time
            
            found, score, matched_ref, matched_doc = check_refs_in_results(retrieved_docs, expected_refs)
            
            query_result = {
                "query": query,
                "case": case_name,
                "top_k": top_k,
                "expected_refs": expected_refs,
                "search_time_seconds": round(search_time, 4),
                "retrieved_docs_count": len(retrieved_docs),
                "check": {
                    "found": found,
                    "match_score": score if found else None,
                    "matched_ref": matched_ref if found else None,
                    "matched_doc": matched_doc if found else None
                }
            }
            results.append(query_result)
        
    return results

def save_results_to_json(results, output_path, test_file):
    """Saves the test results to a JSON file with metadata."""
    
    # Group results by case
    case_results = {}
    for r in results:
        case = r.get("case", "unknown")
        if case not in case_results:
            case_results[case] = []
        case_results[case].append(r)
    
    summary = {
        "test_file": test_file,
        "test_timestamp": datetime.now().isoformat(),
        "total_queries_processed": len(results) // len(case_results) if case_results else 0,  # per case
    }
    
    for case, res_list in case_results.items():
        top_k = res_list[0].get("top_k", 7) if res_list else 7
        valid_test_cases = [r for r in res_list if r.get("expected_refs")]
        total_valid = len(valid_test_cases)
        if total_valid > 0:
            hits = sum(1 for r in valid_test_cases if r["check"]["found"])
            summary[f"total_valid_queries_for_hit_rate_{case}"] = total_valid
            summary[f"hit_rate_top_{top_k}_{case}"] = f"{hits}/{total_valid} ({hits/total_valid:.2%})"
        else:
            summary[f"total_valid_queries_for_hit_rate_{case}"] = 0
            summary[f"hit_rate_top_{top_k}_{case}"] = "0/0 (No valid queries)"
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nTest results saved to: {output_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

def main():
    """Main function to run the automated testing script."""
    parser = argparse.ArgumentParser(description="Automated testing for chatbot retrieval with hybrid search and reranking.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input Excel file (e.g., test_data.xlsx).")
    parser.add_argument("-o", "--output", default="test_results_hybrid.json", help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    try:
        test_data = pd.read_excel(args.input, header=1, engine='openpyxl')
        # Remove empty rows
        test_data = test_data.dropna(how='all').reset_index(drop=True)
        if 'query' not in test_data.columns:
            raise ValueError("The input Excel file must contain a 'query' column.")
        if 'answer' not in test_data.columns:
            for alt in ['expected', 'ground_truth', 'reference', 'refs']:
                if alt in test_data.columns:
                    test_data['answer'] = test_data[alt]
                    break
            if 'answer' not in test_data.columns:
                raise ValueError("The input Excel file must contain an 'answer' column with the chatbot answer text including 'Cơ sở pháp lý:'.")

    except FileNotFoundError:
        print(f"Error: The file '{args.input}' was not found.")
        return
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    print("Initializing clients...")
    try:
        qdrant_client, embedding_model = initialize_clients()
    except ValueError as e:
        print(f"Initialization failed: {e}")
        return
    
    # Initialize global BM25
    global all_docs, bm25_global
    print("Loading all documents for BM25...")
    all_docs = load_all_docs(qdrant_client)
    tokenized_corpus = [tokenize(d['content']) for d in all_docs]
    bm25_global = BM25Okapi(tokenized_corpus)
    print(f"Loaded {len(all_docs)} documents for BM25.")

    print(f"Starting test with {len(test_data)} queries from '{args.input}'...")
    test_results = run_test(qdrant_client, embedding_model, test_data)
    
    save_results_to_json(test_results, args.output, args.input)

if __name__ == "__main__":
    main()
