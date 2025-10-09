
import os
import json
import time
import argparse
import re
import numpy as np
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# ================== CONFIG ==================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ================== INITIALIZATION ==================
def initialize_clients():
    """Initializes and returns Qdrant and SentenceTransformer clients."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file.")
    
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return qdrant_client, embedding_model

def embed_query(model, query):
    """Embeds a single query using the provided SentenceTransformer model."""
    return model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

def search_qdrant(client, vector, top_k):
    """Performs a search on Qdrant with the given vector and top_k."""
    try:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )
        return search_result
    except Exception as e:
        print(f"An error occurred during Qdrant search: {e}")
        return []

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
            refs.append({"article": int(article), "clause": int(clause), "point": point.lower()})
            continue
        m = patt2.search(p)
        if m:
            clause, article, point = m.group(1), m.group(2), m.group(3)
            refs.append({
                "article": int(article),
                "clause": int(clause) if clause else None,
                "point": point.lower() if point else None,
            })
            continue
        m = patt3.search(p)
        if m:
            article, clause, point = m.group(1), m.group(2), m.group(3)
            refs.append({
                "article": int(article),
                "clause": int(clause) if clause else None,
                "point": point.lower() if point else None,
            })

    # Deduplicate
    uniq = []
    seen = set()
    for r in refs:
        key = (r.get("article"), r.get("clause"), r.get("point"))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq

def _result_payload_triplet(result):
    pld = result.payload if hasattr(result, 'payload') else (result.get('payload') if isinstance(result, dict) else {})
    art = pld.get('article_no')
    cls = pld.get('clause_no')
    pt  = pld.get('point_letter')
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

def run_test(qdrant_client, embedding_model, test_data, top_k_values):
    """Runs the automated test for each query in the test data."""
    results = []
    
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing queries"):
        query = row['query']
        # 'answer' is expected to contain the chatbot answer text, including 'Cơ sở pháp lý: ...'
        answer_text = row.get('answer')
        expected_refs = extract_legal_refs(answer_text) if isinstance(answer_text, str) else []
        
        # Debugging for the first 5 queries
        if index < 5:
            print("\n" + "="*20 + f" DEBUG: Query #{index+1} " + "="*20)
            print(f"Query: {query}")
            print(f"Extracted expected refs: {expected_refs}")

        start_time = time.time()
        query_vector = embed_query(embedding_model, query)
        embedding_time = time.time() - start_time
        
        query_result = {
            "query": query,
            "expected_refs": expected_refs,
            "embedding_time_seconds": round(embedding_time, 4),
            "checks": {}
        }
        
        for k in sorted(top_k_values):
            start_time = time.time()
            search_results = search_qdrant(qdrant_client, query_vector, k)
            search_time = time.time() - start_time
            
            # Debugging for the first 5 queries
            if index < 5:
                retrieved_articles = [res.payload.get('article_no') for res in search_results]
                print(f"  - Top {k} Retrieved Articles: {retrieved_articles}")
            found, score, matched_ref, matched_doc = check_refs_in_results(search_results, expected_refs)
            
            query_result["checks"][f"top_{k}"] = {
                "found": found,
                "search_time_seconds": round(search_time, 4),
                "match_score": score if found else None,
                "retrieved_count": len(search_results),
                "matched_ref": matched_ref if found else None,
                "matched_doc": matched_doc if found else None
            }
        
        if index < 4:
             print("="*58)
        elif index == 4:
            print("="*58)
            print("\n[INFO] End of debugging output. Continuing with the rest of the queries...\n")


        results.append(query_result)
        
    return results

def save_results_to_json(results, output_path, test_file, top_k_values):
    """Saves the test results to a JSON file with metadata."""
    
    # Identify valid test cases (where expected refs were extracted)
    valid_test_cases = [r for r in results if r.get("expected_refs")] 
    total_valid_queries = len(valid_test_cases)

    summary = {
        "test_file": test_file,
        "test_timestamp": datetime.now().isoformat(),
        "total_queries_processed": len(results),
        "total_valid_queries_for_hit_rate": total_valid_queries,
        "top_k_values_tested": sorted(top_k_values)
    }
    
    # Calculate hit rates for each top_k based on valid queries only
    for k in sorted(top_k_values):
        if total_valid_queries > 0:
            hits = sum(1 for r in valid_test_cases if r["checks"][f"top_{k}"]["found"])
            summary[f"hit_rate_top_{k}"] = f"{hits}/{total_valid_queries} ({hits/total_valid_queries:.2%})"
        else:
            summary[f"hit_rate_top_{k}"] = "0/0 (No valid queries to calculate hit rate)"
        
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
    parser = argparse.ArgumentParser(description="Automated testing for chatbot retrieval.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input Excel file (e.g., test_data.xlsx).")
    parser.add_argument("-o", "--output", default="test_results.json", help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    try:
        # Keep header=1 as before to align with the sheet structure; adjust if needed later
        test_data = pd.read_excel(args.input, header=1)
        # Ensure columns
        if 'query' not in test_data.columns:
            raise ValueError("The input Excel file must contain a 'query' column.")
        if 'answer' not in test_data.columns:
            # Try alternative column names
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

    top_k_values = [10, 15, 20]
    
    print("Initializing clients...")
    try:
        qdrant_client, embedding_model = initialize_clients()
    except ValueError as e:
        print(f"Initialization failed: {e}")
        return
    
    print(f"Starting test with {len(test_data)} queries from '{args.input}'...")
    test_results = run_test(qdrant_client, embedding_model, test_data, top_k_values)
    
    save_results_to_json(test_results, args.output, args.input, top_k_values)

if __name__ == "__main__":
    main()
