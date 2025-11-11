import os
import glob
import json
import re
import statistics
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Set, Tuple, Any, Dict, List

# --- Cấu hình và Hằng số ---

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
#COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ten_collection_cua_ban")
COLLECTION_NAME = "hybrid-BDS"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

DENSE_VECTOR_NAME = "bge-m3"
SPARSE_VECTOR_NAME = "bm25"
BM25_SPARSE_MODEL_NAME = "Qdrant/bm25"
RRF_RANK_CONST = 60

# Weighted fusion cho RRF (có thể tune)
WEIGHT_DENSE = 1.2
WEIGHT_SPARSE = 0.8

DATA_FOLDER = "data"
TEST_DATA_FILE = "BDS_Test.xlsx"
OUTPUT_FILE = "results/results_BAAI_BDS_HYBRID_V1.json"
TOP_K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
MAX_K = max(TOP_K_VALUES)

# --- Các Hàm Tiện Ích ---

def initialize_clients() -> Tuple[SentenceTransformer, QdrantClient]:
    """Khởi tạo mô hình embedding và Qdrant client."""
    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Đang kết nối tới Qdrant tại: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Kết nối Qdrant và collection '{COLLECTION_NAME}' thành công.")
    except Exception as e:
        print(f"LỖI: Không thể kết nối hoặc tìm thấy collection '{COLLECTION_NAME}'.")
        print(f"Chi tiết lỗi: {e}")
        raise
        
    return model, client

def load_excel_data(folder_path: str, specific_filename: str) -> pd.DataFrame:
    """Tải và xác thực file Excel được chỉ định từ thư mục."""
    
    if not specific_filename:
        print(f"LỖI: Biến môi trường 'TEST_DATA_FILE' chưa được set trong file .env.")
        raise ValueError("Chưa chỉ định file test data.")
        
    file_path = os.path.join(folder_path, specific_filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file được chỉ định: {file_path}")

    print(f"Đang đọc file: {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.title()
    
    if "Query" not in df.columns or "Positive" not in df.columns:
        raise ValueError("File Excel phải chứa 2 cột bắt buộc: 'Query' và 'Positive'.")
        
    initial_rows = len(df)
    df.dropna(subset=["Query", "Positive"], how='all', inplace=True)
    df.dropna(subset=["Query", "Positive"], how='any', inplace=True)
    final_rows = len(df)
    
    print(f"Đã tải {final_rows} hàng hợp lệ (loại bỏ {initial_rows - final_rows} hàng trống/không hợp lệ).")
    return df

def parse_references(text: str) -> Set[Tuple[str, str, str]]:
    """
    Trích xuất các tham chiếu (Điều, Khoản, Điểm) từ văn bản "Positive".
    """
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
        khoan_raw = khoan_match.group(1) if khoan_match else None
        diem_raw = diem_match.group(1) if diem_match else None

        khoan = None
        if khoan_raw:
            khoan_num_match = re.match(r"^(\d+)", khoan_raw)
            if khoan_num_match:
                khoan = khoan_num_match.group(1)
            else:
                khoan = khoan_raw

        diem = None
        if diem_raw:
            diem = re.sub(r"[^a-zA-ZđĐ]", "", diem_raw).lower() or None

        if dieu or khoan or diem:
            references.add((dieu, khoan, diem))
            
    return references

def normalize_payload_ref(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Chuẩn hóa payload từ Qdrant thành một tuple (Điều, Khoản, Điểm).
    """
    if not isinstance(payload, dict):
        return (None, None, None)

    meta = None
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        meta = payload["metadata"]
    else:
        meta = payload

    d = meta.get("article_no")
    k_raw = meta.get("clause_no")
    p = meta.get("point_letter") or meta.get("point_id") or None

    d_str = str(d) if d is not None else None

    k_str = None
    if k_raw is not None:
        k_raw_str = str(k_raw).strip()
        k_num_match = re.match(r"^(\d+)", k_raw_str)
        if k_num_match:
            k_str = k_num_match.group(1)
        else:
            k_str = k_raw_str

    if p is None:
        p_str = None
    else:
        p_str = str(p).strip()
        p_str = re.sub(r"[^a-zA-ZđĐ]", "", p_str)
        p_str = p_str.lower() if p_str else None

    return (d_str, k_str, p_str)

def ground_truth_matches(retrieved_ref: Tuple[str,str,str], gt_ref: Tuple[str,str,str]) -> bool:
    """
    Trả về True nếu retrieved_ref khớp với gt_ref.
    QUY TẮC: Chỉ kiểm tra Điều và Điểm. Khoản (clause) bị BỎ QUA.
    """
    rd, _rk, rp = retrieved_ref
    gd, _gk, gp = gt_ref

    if gd is not None:
        if gd != rd:
            return False

    if gp is not None:
        if gp != rp:
            return False
            
    return True

def merge_hit_payloads(hit1, hit2):
    """
    Merge payloads từ 2 hits của cùng 1 document.
    Ưu tiên metadata đầy đủ hơn.
    """
    payload1 = hit1.payload if hasattr(hit1, 'payload') else {}
    payload2 = hit2.payload if hasattr(hit2, 'payload') else {}
    
    if not payload1:
        payload1 = {}
    if not payload2:
        payload2 = {}
    
    # Merge: payload1 overrides payload2
    merged = {**payload2, **payload1}
    hit1.payload = merged
    return hit1

def reciprocal_rank_fusion(
    dense_results, 
    sparse_results, 
    k: int = 60,
    weight_dense: float = 1.0,
    weight_sparse: float = 1.0,
    normalize: bool = True
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    RRF chuẩn với normalization và weighted fusion.
    
    Args:
        dense_results: Kết quả từ dense vector search
        sparse_results: Kết quả từ sparse (BM25) search
        k: Hằng số RRF (thường là 60)
        weight_dense: Trọng số cho dense results
        weight_sparse: Trọng số cho sparse results
        normalize: Có normalize scores về [0, 1] không
    
    Returns:
        fused_scores: Dict[id -> normalized_score]
        all_hits_map: Dict[id -> ScoredPoint với merged payload]
    """
    fused_scores = {}
    all_hits_map = {}
    
    # Process dense results
    for rank, hit in enumerate(dense_results.points, start=1):
        all_hits_map[hit.id] = hit
        score = weight_dense / (k + rank)
        fused_scores[hit.id] = score
    
    # Process sparse results
    for rank, hit in enumerate(sparse_results.points, start=1):
        score = weight_sparse / (k + rank)
        
        if hit.id in all_hits_map:
            # Merge payloads để không mất metadata
            all_hits_map[hit.id] = merge_hit_payloads(all_hits_map[hit.id], hit)
            # Cộng sparse score vào
            fused_scores[hit.id] += score
        else:
            all_hits_map[hit.id] = hit
            fused_scores[hit.id] = score
    
    # Normalize scores về [0, 1]
    if normalize and fused_scores:
        max_score = max(fused_scores.values())
        min_score = min(fused_scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            for doc_id in fused_scores:
                fused_scores[doc_id] = (fused_scores[doc_id] - min_score) / score_range
    
    return fused_scores, all_hits_map

def calculate_mrr(first_hit_ranks: List[int]) -> float:
    """
    Tính Mean Reciprocal Rank (MRR).
    MRR = Average(1 / rank của first hit)
    """
    if not first_hit_ranks:
        return 0.0
    reciprocal_ranks = [1.0 / rank for rank in first_hit_ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def calculate_precision_at_k(
    retrieved_refs: List[Tuple[str, str, str]], 
    ground_truth_refs: Set[Tuple[str, str, str]],
    k: int
) -> float:
    """
    Tính Precision@K.
    Precision@K = (Số docs đúng trong top K) / K
    """
    retrieved_at_k = retrieved_refs[:k]
    correct = sum(
        1 for rr in retrieved_at_k
        if any(ground_truth_matches(rr, gt) for gt in ground_truth_refs)
    )
    return correct / k if k > 0 else 0.0

def run_test(
    query: str, 
    ground_truth_refs: Set[Tuple[str, str, str]], 
    model: SentenceTransformer, 
    client: QdrantClient
) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]], Any, Any, Dict[int, float]]:
    """
    Thực hiện embedding, retrieval (HYBRID + RRF V2) và so sánh.
    
    Returns:
        hits_at_k: Dict với hit/miss cho từng K
        found_in_max_k: Boolean - có tìm thấy trong top MAX_K không
        retrieved_payloads: List các payload đã retrieve
        first_hit_rank: Rank của first hit (hoặc None)
        first_hit_score: Score của first hit (hoặc None)
        precision_at_k: Dict với precision cho từng K
    """
    
    # 0. Clean text
    cleaned_query = query.replace('\x00', '').replace('\u200b', '')
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    # 1. Embedding (Dense)
    query_vector = model.encode(cleaned_query, convert_to_tensor=False).tolist()

    # 2. Retrieval - Dense
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=MAX_K,
        with_payload=True
    )
    
    # 3. Retrieval - Sparse (BM25)
    sparse_query = models.Document(
        text=cleaned_query,
        model=BM25_SPARSE_MODEL_NAME
    )
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_query,
        using=SPARSE_VECTOR_NAME,
        limit=MAX_K,
        with_payload=True
    )
    
    # 4. Reciprocal Rank Fusion (RRF V2 - FIXED)
    fused_scores, all_hits_map = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=RRF_RANK_CONST,
        weight_dense=WEIGHT_DENSE,
        weight_sparse=WEIGHT_SPARSE,
        normalize=True
    )

    # 5. Sắp xếp theo RRF score
    sorted_hit_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    
    # 6. Tạo danh sách kết quả cuối cùng với RRF score
    top_max_results = []
    for hit_id in sorted_hit_ids[:MAX_K]:
        hit = all_hits_map[hit_id]
        hit.score = fused_scores[hit_id]  # Normalized RRF score
        top_max_results.append(hit)

    # 7. Chuẩn hóa kết quả retrieve
    def compact_hit(hit) -> Dict[str, Any]:
        payload = hit.payload if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
        meta = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else payload

        art = meta.get('article_no')
        cla = meta.get('clause_no')
        pt = meta.get('point_letter') or meta.get('point_id') or None
        if pt is not None:
            pt = re.sub(r"[^a-zA-ZđĐ]", "", str(pt)).lower() or None
        
        score = getattr(hit, 'score', None)

        return {
            "id": meta.get('id') or payload.get('id'),
            "article_no": str(art) if art is not None else None,
            "clause_no": str(cla) if cla is not None else None,
            "point_letter": pt,
            "article_title": meta.get('article_title'),
            "exact_citation": meta.get('exact_citation'),
            "score": score
        }

    retrieved_payloads = [compact_hit(hit) for hit in top_max_results]
    
    # 8. So sánh với ground truth
    hits_at_k = {}
    found_in_max_k = False
    first_hit_rank = None
    first_hit_score = None
    precision_at_k = {}
    
    retrieved_refs_all = [normalize_payload_ref(hit.payload) for hit in top_max_results]
    retrieved_scores_all = [hit.score for hit in top_max_results]

    # Tìm first hit
    for idx, rr in enumerate(retrieved_refs_all, start=1):
        for gt in ground_truth_refs:
            if ground_truth_matches(rr, gt):
                first_hit_rank = idx
                try:
                    first_hit_score = retrieved_scores_all[idx-1]
                except Exception:
                    first_hit_score = None
                break
        if first_hit_rank is not None:
            break

    # Tính hit@K và precision@K cho từng K
    for k in TOP_K_VALUES:
        retrieved_refs_at_k = retrieved_refs_all[:k]
        
        # Debug (chỉ in 1 lần)
        try:
            if run_test.debug_count < 1:
                print(f"DEBUG Query: {query}")
                print(f"DEBUG Ground truth: {ground_truth_refs}")
                print(f"DEBUG Retrieved (RRF V2) at {k}: {retrieved_refs_at_k}")
                run_test.debug_count += 1
        except AttributeError:
            run_test.debug_count = 1
            print(f"DEBUG Query: {query}")
            print(f"DEBUG Ground truth: {ground_truth_refs}")
            print(f"DEBUG Retrieved (RRF V2) at {k}: {retrieved_refs_at_k}")
        
        # Hit@K
        is_hit = False
        for gt in ground_truth_refs:
            for rr in retrieved_refs_at_k:
                if ground_truth_matches(rr, gt):
                    is_hit = True
                    break
            if is_hit:
                break
        
        hits_at_k[f"hit_at_{k}"] = is_hit
        if is_hit:
            found_in_max_k = True
        
        # Precision@K
        precision_at_k[k] = calculate_precision_at_k(
            retrieved_refs_all,
            ground_truth_refs,
            k
        )
            
    return hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score, precision_at_k

def run_ablation_study(
    query: str,
    ground_truth_refs: Set[Tuple[str, str, str]],
    model: SentenceTransformer,
    client: QdrantClient,
    k: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Chạy ablation study: so sánh Dense-only, Sparse-only, Hybrid RRF.
    
    Returns:
        Dict với keys: "dense_only", "sparse_only", "hybrid_rrf"
        Mỗi value chứa: {"hit": bool, "first_rank": int or None}
    """
    results = {}
    
    cleaned_query = query.replace('\x00', '').replace('\u200b', '')
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    query_vector = model.encode(cleaned_query, convert_to_tensor=False).tolist()
    
    # 1. Dense only
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=k,
        with_payload=True
    )
    dense_refs = [normalize_payload_ref(hit.payload) for hit in dense_results.points]
    dense_hit = any(
        ground_truth_matches(rr, gt)
        for gt in ground_truth_refs
        for rr in dense_refs
    )
    dense_first_rank = None
    for idx, rr in enumerate(dense_refs, start=1):
        if any(ground_truth_matches(rr, gt) for gt in ground_truth_refs):
            dense_first_rank = idx
            break
    results["dense_only"] = {"hit": dense_hit, "first_rank": dense_first_rank}
    
    # 2. Sparse only
    sparse_query = models.Document(text=cleaned_query, model=BM25_SPARSE_MODEL_NAME)
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_query,
        using=SPARSE_VECTOR_NAME,
        limit=k,
        with_payload=True
    )
    sparse_refs = [normalize_payload_ref(hit.payload) for hit in sparse_results.points]
    sparse_hit = any(
        ground_truth_matches(rr, gt)
        for gt in ground_truth_refs
        for rr in sparse_refs
    )
    sparse_first_rank = None
    for idx, rr in enumerate(sparse_refs, start=1):
        if any(ground_truth_matches(rr, gt) for gt in ground_truth_refs):
            sparse_first_rank = idx
            break
    results["sparse_only"] = {"hit": sparse_hit, "first_rank": sparse_first_rank}
    
    # 3. Hybrid RRF (lấy từ run_test, nhưng chỉ check top k)
    hits_at_k, _, _, first_hit_rank, _, _ = run_test(query, ground_truth_refs, model, client)
    hybrid_hit = hits_at_k.get(f"hit_at_{k}", False)
    hybrid_first_rank = first_hit_rank if first_hit_rank and first_hit_rank <= k else None
    results["hybrid_rrf"] = {"hit": hybrid_hit, "first_rank": hybrid_first_rank}
    
    return results

# --- Hàm Chính ---

def main():
    """Hàm thực thi chính của quy trình test."""
    try:
        model, client = initialize_clients()
        df = load_excel_data(DATA_FOLDER, TEST_DATA_FILE)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi khởi tạo hoặc tải dữ liệu: {e}")
        return

    total_queries = len(df)
    
    # Khởi tạo cấu trúc báo cáo
    summary = {
        "test_type": "Hybrid RRF V2 (Fixed + Enhanced)",
        "dense_vector": DENSE_VECTOR_NAME,
        "sparse_vector": SPARSE_VECTOR_NAME,
        "bm25_model": BM25_SPARSE_MODEL_NAME,
        "rrf_k_const": RRF_RANK_CONST,
        "weight_dense": WEIGHT_DENSE,
        "weight_sparse": WEIGHT_SPARSE,
        "normalization_enabled": True,
        "total_queries_in_file": total_queries,
        "scanned_queries": 0,
        "queries_with_no_hit": 0,
    }
    summary.update({f"hit_at_{k}": 0 for k in TOP_K_VALUES})
    summary.update({f"avg_precision_at_{k}": 0.0 for k in TOP_K_VALUES})
    
    # Ablation study summary
    ablation_summary = {
        "dense_only": {"total_hits": 0, "avg_first_rank": []},
        "sparse_only": {"total_hits": 0, "avg_first_rank": []},
        "hybrid_rrf": {"total_hits": 0, "avg_first_rank": []}
    }
    
    missed_queries_details = []
    first_hit_ranks: List[int] = []
    queries_with_any_first_hit = 0
    first_hit_scores: List[float] = []
    
    # Precision tracking
    precision_sums = {k: 0.0 for k in TOP_K_VALUES}

    print(f"\n--- BẮT ĐẦU QUÁ TRÌNH TEST HYBRID RRF V2 ({total_queries} QUERIES) ---")

    # Lặp qua từng hàng trong DataFrame
    for index, row in df.iterrows():
        query = row["Query"]
        positive_text = row["Positive"]
        
        try:
            # 1. Lấy ground truth
            ground_truth_refs = parse_references(positive_text)
            
            if not ground_truth_refs:
                print(f"Cảnh báo: Không thể trích xuất tham chiếu từ 'Positive' cho query: '{query}'. Bỏ qua.")
                continue

            # 2. Chạy test chính
            hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score, precision_at_k = run_test(
                query, ground_truth_refs, model, client
            )
            
            # 3. Chạy ablation study (mỗi 10 queries để tiết kiệm thời gian)
            if summary["scanned_queries"] % 10 == 0:
                ablation_results = run_ablation_study(query, ground_truth_refs, model, client, k=10)
                for method, metrics in ablation_results.items():
                    if metrics["hit"]:
                        ablation_summary[method]["total_hits"] += 1
                    if metrics["first_rank"] is not None:
                        ablation_summary[method]["avg_first_rank"].append(metrics["first_rank"])
            
            # 4. Cập nhật thống kê
            summary["scanned_queries"] += 1
            for k_str, is_hit in hits_at_k.items():
                if is_hit:
                    summary[k_str] += 1
            
            # Cập nhật precision
            for k, prec in precision_at_k.items():
                precision_sums[k] += prec
                    
            # 5. Ghi lại các trường hợp "miss"
            if not found_in_max_k:
                summary["queries_with_no_hit"] += 1
                missed_queries_details.append({
                    "query": query,
                    "expected_references": [str(ref) for ref in ground_truth_refs],
                    f"retrieved_top_{MAX_K}": retrieved_payloads
                })
            
            # 6. Record first hit stats
            if first_hit_rank is not None:
                first_hit_ranks.append(first_hit_rank)
                queries_with_any_first_hit += 1
            if first_hit_score is not None:
                try:
                    first_hit_scores.append(float(first_hit_score))
                except Exception:
                    pass

            if (summary["scanned_queries"] % 10) == 0:
                print(f"Đã xử lý {summary['scanned_queries']}/{total_queries} queries...")

        except Exception as e:
            print(f"Lỗi khi xử lý query: '{query}'. Lỗi: {e}")
            missed_queries_details.append({
                "query": query,
                "error": str(e)
            })

    print("--- HOÀN TẤT QUÁ TRÌNH TEST ---")
    
    # Tính toán recall & precision
    for k in TOP_K_VALUES:
        hit_count = summary[f"hit_at_{k}"]
        recall = (hit_count / summary["scanned_queries"]) * 100 if summary["scanned_queries"] > 0 else 0
        summary[f"recall_at_{k}_percent"] = round(recall, 2)
        
        avg_precision = precision_sums[k] / summary["scanned_queries"] if summary["scanned_queries"] > 0 else 0
        summary[f"avg_precision_at_{k}"] = round(avg_precision, 4)

    # Tính MRR
    mrr = calculate_mrr(first_hit_ranks)
    summary["mrr"] = round(mrr, 4)
    
    # Tính thống kê first-hit
    if first_hit_ranks:
        avg_first_hit = sum(first_hit_ranks) / len(first_hit_ranks)
    else:
        avg_first_hit = None

    summary["avg_first_hit_rank"] = round(avg_first_hit, 2) if avg_first_hit is not None else None
    summary["pct_queries_with_first_hit_within_top_{0}".format(MAX_K)] = round((queries_with_any_first_hit / summary["scanned_queries"])*100, 2) if summary["scanned_queries"]>0 else 0.0
    
    # Thống kê điểm RRF
    if first_hit_scores:
        avg_first_hit_score = statistics.mean(first_hit_scores)
        std_first_hit_score = statistics.pstdev(first_hit_scores)
        max_first_hit_score = max(first_hit_scores)
        min_first_hit_score = min(first_hit_scores)
    else:
        avg_first_hit_score = None
        std_first_hit_score = None
        max_first_hit_score = None
        min_first_hit_score = None

    summary["avg_first_hit_score"] = round(avg_first_hit_score, 4) if avg_first_hit_score is not None else None
    summary["std_first_hit_score"] = round(std_first_hit_score, 4) if std_first_hit_score is not None else None
    summary["max_first_hit_score"] = round(max_first_hit_score, 4) if max_first_hit_score is not None else None
    summary["min_first_hit_score"] = round(min_first_hit_score, 4) if min_first_hit_score is not None else None
    
    if avg_first_hit_score is not None and std_first_hit_score is not None:
        suggested = avg_first_hit_score - std_first_hit_score
        summary["suggested_score_threshold_mean_minus_std"] = round(suggested, 4)
    else:
        summary["suggested_score_threshold_mean_minus_std"] = None
    
    # Finalize ablation study stats
    for method in ablation_summary:
        total_tested = summary["scanned_queries"] // 10  # Tested every 10 queries
        if total_tested > 0:
            ablation_summary[method]["recall@10_percent"] = round(
                (ablation_summary[method]["total_hits"] / total_tested) * 100, 2
            )
        else:
            ablation_summary[method]["recall@10_percent"] = 0.0
        
        if ablation_summary[method]["avg_first_rank"]:
            ablation_summary[method]["avg_first_rank"] = round(
                sum(ablation_summary[method]["avg_first_rank"]) / len(ablation_summary[method]["avg_first_rank"]), 2
            )
        else:
            ablation_summary[method]["avg_first_rank"] = None

    # 5. Tạo báo cáo cuối cùng
    report = {
        "summary": summary,
        "ablation_study": ablation_summary,
        "missed_queries_details": missed_queries_details
    }
    
    # 6. Lưu file JSON
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print(f"\nBáo cáo đã được lưu thành công tại: {OUTPUT_FILE}")
        
        print("\n--- TÓM TẮT KẾT QUẢ HYBRID RRF V2 (FIXED) ---")
        print(json.dumps(summary, indent=4, ensure_ascii=False))
        
        print("\n--- ABLATION STUDY (Dense vs Sparse vs Hybrid) ---")
        print(json.dumps(ablation_summary, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"LỖI: Không thể ghi file báo cáo {OUTPUT_FILE}. Lỗi: {e}")

if __name__ == "__main__":
    main()