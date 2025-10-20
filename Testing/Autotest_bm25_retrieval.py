import os
import re
import json
import statistics
from typing import Set, Tuple, Any, Dict, List

from dotenv import load_dotenv
import pandas as pd
import numpy as np

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# BM25
from rank_bm25 import BM25Okapi

# --- Cấu hình và hằng số ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

DATA_FOLDER = "data"
TEST_DATA_FILE = "HNGD_Test.xlsx"
OUTPUT_FILE = "results/results_minhquan_bm25_retrieval.json"
TOP_K_VALUES = [5, 10, 15, 20, 25, 30, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
MAX_K = max(TOP_K_VALUES)

# Hybrid weight: alpha for vector, (1-alpha) for BM25
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.6))

# --- Utils / parsing (tương tự Autotest.py) ---
def parse_references(text: str) -> Set[Tuple[str, str, str]]:
    references = set()
    parts = re.split(r'\s+(?:và|hoặc)\s+', str(text).strip(), flags=re.IGNORECASE)
    for part in parts:
        part = part.strip().rstrip(',.')
        if not part:
            continue
        dieu_match = re.search(r"Điều\s+(\w+)", part, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\w+)", part, re.IGNORECASE)
        diem_match = re.search(r"Điểm\s+(\w+)", part, re.IGNORECASE)
        dieu = dieu_match.group(1) if dieu_match else None
        khoan = khoan_match.group(1) if khoan_match else None
        diem = diem_match.group(1) if diem_match else None
        if dieu or khoan or diem:
            references.add((dieu, khoan, diem))
    return references

def normalize_payload_ref(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    if not isinstance(payload, dict):
        return (None, None, None)
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
    d = meta.get("article_no")
    k = meta.get("clause_no")
    p = meta.get("point_letter") or meta.get("point_id") or None
    d_str = str(d) if d is not None else None
    k_str = str(k) if k is not None else None
    if p is None:
        p_str = None
    else:
        p_str = str(p).strip()
        p_str = re.sub(r"[^a-zA-ZđĐ]", "", p_str)
        p_str = p_str.lower() if p_str else None
    return (d_str, k_str, p_str)

# --- Qdrant + BM25 Corpus building ---
def initialize_clients() -> Tuple[SentenceTransformer, QdrantClient]:
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Connecting to Qdrant: {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Connected to collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"ERROR connecting to Qdrant collection '{COLLECTION_NAME}': {e}")
        raise
    return model, client

def _make_doc_text(payload: Dict[str, Any]) -> str:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
    parts = []
    for k in ("article_title", "exact_citation", "text", "content", "law_text"):
        v = meta.get(k)
        if v:
            parts.append(str(v))
    # Also include article/clause/point tokens
    for k in ("article_no", "clause_no", "point_letter"):
        v = meta.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts)

def build_corpus_from_qdrant(client: QdrantClient) -> List[Dict[str, Any]]:
    print("Building corpus from Qdrant (scrolling payloads)...")
    docs = []
    limit = 100
    next_offset = None # Bắt đầu từ offset đầu tiên

    while True:
        # Client trả về (list_of_points, next_offset)
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=next_offset, # Sử dụng offset của trang trước
            with_payload=True
        )

        if not points:
            break # Không còn điểm nào

        for p in points:
            # Client v1.x trả về Pydantic object, không phải dict
            payload = p.payload if hasattr(p, 'payload') else {}
            doc_id = p.id if hasattr(p, 'id') else None
            
            doc_text = _make_doc_text(payload)
            docs.append({
                "id": doc_id,
                "payload": payload,
                "text": doc_text
            })

        if next_offset is None:
            break # Qdrant báo đây là trang cuối cùng

    print(f"Collected {len(docs)} documents for BM25 index.")
    return docs

def tokenize(text: str) -> List[str]:
    # simple tokenization: extract words (keeps Vietnamese letters)
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text, flags=re.UNICODE)
    return [t.lower() for t in tokens if t.strip()]

def build_bm25_index(docs: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    corpus_tokens = [tokenize(doc["text"] or "") for doc in docs]
    
    # KIỂM TRA QUAN TRỌNG: Lọc bỏ các tài liệu rỗng
    non_empty_corpus_tokens = [tokens for tokens in corpus_tokens if tokens] # Chỉ giữ lại list nào không rỗng
    
    if not non_empty_corpus_tokens:
        # Nếu không còn tài liệu nào, không thể xây dựng BM25
        print("ERROR: Corpus is empty after tokenization. Cannot build BM25 index.")
        # Bạn có thể raise lỗi ở đây hoặc trả về None
        raise ValueError("Cannot build BM25 index from an empty corpus.")
        
    print(f"Building BM25 index from {len(non_empty_corpus_tokens)} non-empty documents (out of {len(docs)} total).")
    bm25 = BM25Okapi(non_empty_corpus_tokens)
    return bm25, corpus_tokens # Vẫn trả về corpus_tokens gốc nếu bạn cần nó để map index

# --- Hybrid search ---
def hybrid_search(query: str, model: SentenceTransformer, client: QdrantClient,
                  bm25: BM25Okapi, docs: List[Dict[str, Any]],
                  top_k_vector: int = MAX_K, top_k_bm25: int = MAX_K,
                  alpha: float = HYBRID_ALPHA) -> List[Dict[str, Any]]:
    # 1) vector search
    q_vec = model.encode(query, convert_to_tensor=False).tolist()
    vec_results = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=top_k_vector, with_payload=True)
    vec_map = {}
    vec_scores = []
    for hit in vec_results:
        # support both object and dict responses
        payload = getattr(hit, "payload", hit.get("payload") if isinstance(hit, dict) else {})
        pid = getattr(hit, "id", hit.get("id") if isinstance(hit, dict) else None)
        score = getattr(hit, "score", None)
        # try distance fallback
        if score is None:
            score = payload.get("_qdrant_score") if isinstance(payload, dict) else None
        if score is None:
            # last resort: give small positive score
            score = 0.0
        vec_map[pid] = {"payload": payload, "vec_score": float(score)}
        vec_scores.append(float(score) if score is not None else 0.0)

    # 2) BM25 scores
    q_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    bm25_map = {}
    bm25_score_list = []
    for idx in top_bm25_idx:
        doc = docs[idx]
        pid = doc["id"]
        score = float(bm25_scores[idx])
        bm25_map[pid] = {"payload": doc["payload"], "bm25_score": score}
        bm25_score_list.append(score)

    # 3) Combine ids
    all_ids = set(list(vec_map.keys()) + list(bm25_map.keys()))
    # normalize scores
    def _normalize(arr):
        if not arr:
            return {}
        mn = min(arr)
        mx = max(arr)
        if mx - mn <= 1e-9:
            return {i: 1.0 for i in arr}  # identical -> return 1
        norm = {}
        for i, v in enumerate(arr):
            # Map original order index -> normalized value not needed here
            pass
        return None  # not used

    # Build arrays for normalization
    vec_vals = [vec_map[pid]["vec_score"] for pid in vec_map.keys()]
    bm_vals = [bm25_map[pid]["bm25_score"] for pid in bm25_map.keys()]

    def _min_max_norm(value, arr):
        if not arr:
            return 0.0
        mn = min(arr)
        mx = max(arr)
        if mx - mn <= 1e-9:
            return 1.0
        return (value - mn) / (mx - mn)

    combined = []
    for pid in all_ids:
        vec_score = vec_map.get(pid, {}).get("vec_score", None)
        bm_score = bm25_map.get(pid, {}).get("bm25_score", None)
        # normalize each with available arrays
        norm_vec = _min_max_norm(vec_score, vec_vals) if vec_score is not None else 0.0
        norm_bm = _min_max_norm(bm_score, bm_vals) if bm_score is not None else 0.0
        combined_score = alpha * norm_vec + (1 - alpha) * norm_bm
        payload = vec_map.get(pid, {}).get("payload") or bm25_map.get(pid, {}).get("payload") or {}
        combined.append({
            "id": pid,
            "payload": payload,
            "vec_score": vec_score,
            "bm25_score": bm_score,
            "combined_score": combined_score
        })
    # sort by combined_score desc
    combined_sorted = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    return combined_sorted[:MAX_K]

# --- Evaluation runner (adapted from Autotest.py) ---
def run_test(query: str,
             ground_truth_refs: Set[Tuple[str, str, str]],
             model: SentenceTransformer,
             client: QdrantClient,
             bm25: BM25Okapi,
             docs: List[Dict[str, Any]]) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]], Any, Any]:
    # hybrid retrieval
    hybrid_results = hybrid_search(query, model, client, bm25, docs, alpha=HYBRID_ALPHA)

    # prepare retrieved refs
    retrieved_refs_all = [normalize_payload_ref(hit.get("payload", {})) for hit in hybrid_results]

    def ground_truth_matches(retrieved_ref, gt_ref):
        rd, rk, rp = retrieved_ref
        gd, gk, gp = gt_ref
        if gd is not None and rd is not None:
            if gd != rd:
                return False
        elif gd is not None and rd is None:
            return False
        if gk is not None:
            if rk is None:
                return False
            if gk != rk:
                return False
        if gp is not None:
            if rp is None:
                return False
            if gp != rp:
                return False
        return True

    hits_at_k = {}
    found_in_max_k = False
    first_hit_rank = None
    first_hit_score = None

    for idx, rr in enumerate(retrieved_refs_all, start=1):
        for gt in ground_truth_refs:
            if ground_truth_matches(rr, gt):
                if first_hit_rank is None:
                    first_hit_rank = idx
                    first_hit_score = hybrid_results[idx - 1].get("combined_score")
                found_in_max_k = True
                break
        if found_in_max_k and first_hit_rank is not None:
            break

    for k in TOP_K_VALUES:
        top_k_refs = retrieved_refs_all[:k]
        is_hit = False
        for gt in ground_truth_refs:
            for rr in top_k_refs:
                if ground_truth_matches(rr, gt):
                    is_hit = True
                    break
            if is_hit:
                break
        hits_at_k[f"hit_at_{k}"] = is_hit

    # compact payloads for reporting
    retrieved_payloads = [{
        "id": r.get("id"),
        "article_no": r.get("payload", {}).get("article_no") or r.get("payload", {}).get("metadata", {}).get("article_no"),
        "clause_no": r.get("payload", {}).get("clause_no") or r.get("payload", {}).get("metadata", {}).get("clause_no"),
        "point_letter": r.get("payload", {}).get("point_letter") or r.get("payload", {}).get("metadata", {}).get("point_letter"),
        "combined_score": r.get("combined_score")
    } for r in hybrid_results]

    return hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score

# --- Main ---
def load_excel_data(folder_path: str, specific_filename: str) -> pd.DataFrame:
    if not specific_filename:
        raise ValueError("Chưa chỉ định file test data.")
    file_path = os.path.join(folder_path, specific_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl")
    if "Query" not in df.columns or "Positive" not in df.columns:
        raise ValueError("File Excel phải chứa cột 'Query' và 'Positive'.")
    df.dropna(subset=["Query", "Positive"], how='any', inplace=True)
    return df

def main():
    try:
        model, client = initialize_clients()
        df = load_excel_data(DATA_FOLDER, TEST_DATA_FILE)
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    docs = build_corpus_from_qdrant(client)
    if not docs:
        print("No documents extracted from Qdrant; aborting.")
        return
    bm25, corpus_tokens = build_bm25_index(docs)

    summary = {"total_queries_in_file": len(df), "scanned_queries": 0, "queries_with_no_hit": 0}
    summary.update({f"hit_at_{k}": 0 for k in TOP_K_VALUES})

    missed_queries_details = []
    first_hit_ranks = []
    first_hit_scores = []
    queries_with_any_first_hit = 0

    print(f"Starting hybrid BM25+vector evaluation on {len(df)} queries...")
    for _, row in df.iterrows():
        query = row["Query"]
        positive_text = row["Positive"]
        try:
            ground_truth_refs = parse_references(positive_text)
            if not ground_truth_refs:
                print(f"Warning: cannot parse positive refs for query: {query}")
                continue
            hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score = run_test(
                query, ground_truth_refs, model, client, bm25, docs
            )
            summary["scanned_queries"] += 1
            for k_str, is_hit in hits_at_k.items():
                if is_hit:
                    summary[k_str] += 1
            if not found_in_max_k:
                summary["queries_with_no_hit"] += 1
                missed_queries_details.append({
                    "query": query,
                    "expected_references": [str(ref) for ref in ground_truth_refs],
                    "retrieved_top_k": retrieved_payloads
                })
            if first_hit_rank is not None:
                first_hit_ranks.append(first_hit_rank)
                queries_with_any_first_hit += 1
            if first_hit_score is not None:
                first_hit_scores.append(first_hit_score)

        except Exception as e:
            # --- THÊM DÒNG NÀY ĐỂ IN LỖI RA ---
            print(f"!!! LỖI KHI XỬ LÝ QUERY: {query}")
            print(f"!!! NGUYÊN NHÂN: {e}\n")
            
            missed_queries_details.append({"query": query, "error": str(e)})

    # compute recall@k
    for k in TOP_K_VALUES:
        hit_count = summary.get(f"hit_at_{k}", 0)
        recall = (hit_count / summary["scanned_queries"]) * 100 if summary["scanned_queries"] > 0 else 0.0
        summary[f"recall_at_{k}_percent"] = round(recall, 2)

    if first_hit_ranks:
        summary["avg_first_hit_rank"] = round(sum(first_hit_ranks) / len(first_hit_ranks), 2)
    else:
        summary["avg_first_hit_rank"] = None

    summary[f"pct_queries_with_first_hit_within_top_{MAX_K}"] = round((queries_with_any_first_hit / summary["scanned_queries"]) * 100, 2) if summary["scanned_queries"] > 0 else 0.0

    if first_hit_scores:
        summary["avg_first_hit_score"] = round(statistics.mean(first_hit_scores), 4)
        summary["std_first_hit_score"] = round(statistics.pstdev(first_hit_scores), 4) if len(first_hit_scores) > 1 else 0.0
        summary["max_first_hit_score"] = round(max(first_hit_scores), 4)
        summary["min_first_hit_score"] = round(min(first_hit_scores), 4)
    else:
        summary["avg_first_hit_score"] = None
        summary["std_first_hit_score"] = None
        summary["max_first_hit_score"] = None
        summary["min_first_hit_score"] = None

    report = {"summary": summary, "missed_queries_details": missed_queries_details}

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Evaluation finished. Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()