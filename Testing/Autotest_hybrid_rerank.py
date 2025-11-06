import os
import glob
import json
import re
import statistics
import pandas as pd
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import LateInteractionTextEmbedding
from dotenv import load_dotenv
from typing import Set, Tuple, Any, Dict, List, Optional
from itertools import product
# THÊM MỚI: Cần defaultdict để tính RRF
# (KHÔNG CÒN CẦN THIẾT KHI DÙNG PREFETCH)
# from collections import defaultdict 

# --- Cấu hình và Hằng số ---

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")

# --- Cấu hình mô hình và vector ---
BGE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BGE_VECTOR_NAME = "bge-m3"
COLBERT_MODEL_NAME = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
COLBERT_VECTOR_NAME = "colbertv2.0"
BM25_VECTOR_NAME = "bm25"
BM25_SPARSE_MODEL_NAME = "Qdrant/bm25"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cấu hình LƯỚI TEST (GRID SEARCH) ---
DATA_FOLDER = "data"
TEST_DATA_FILE = "HNGD_Full.xlsx"
# THAY ĐỔI: Tên file output mới
OUTPUT_FILE = "results/results_HNGD_V2.json" # Đổi tên file output

K_DENSE_VALUES = [10, 20, 30, 50, 70, 100]
K_SPARSE_VALUES = [10, 20, 30, 50, 70, 100]
FINAL_K_AFTER_RERANK = 7

# --- CẤU HÌNH MỚI CHO PIPELINE CẢI TIẾN ---
# (KHÔNG CẦN CÁC HẰNG SỐ RRF/K_CANDIDATES KHI DÙNG PREFETCH)
# RRF_K_CONST = 60
# K_CANDIDATES_FROM_RRF = 30 
# ----------------------------------------------

# --- HÀM MỚI: Reciprocal Rank Fusion (RRF) ---
# (KHÔNG CẦN THIẾT - QDRANT PREFETCH SẼ THAY THẾ LOGIC NÀY)
# def reciprocal_rank_fusion(...):
#     ...

# --- HÀM MỚI: Tính điểm ColBERT Client-side ---
# (KHÔNG CẦN THIẾT - QDRANT SERVER SẼ TÍNH TOÁN)
# def calculate_colbert_similarity(...):
#     ...

# --- Các Hàm Tiện Ích (Giữ nguyên) ---
def initialize_clients() -> Tuple[SentenceTransformer, LateInteractionTextEmbedding, QdrantClient]:
    print(f"Sử dụng device: {DEVICE}")
    print(f"Đang tải mô hình BGE: {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
    print(f"Đang tải mô hình Colbert (FastEmbed): {COLBERT_MODEL_NAME}...")
    colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL_NAME, device=DEVICE)
    print(f"Đang kết nối tới Qdrant tại: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Kết nối Qdrant và collection '{COLLECTION_NAME}' thành công.")
    except Exception as e:
        print(f"LỖI: Không thể kết nối hoặc tìm thấy collection '{COLLECTION_NAME}'.")
        print(f"Chi tiết lỗi: {e}")
        raise
    return bge_model, colbert_model, client

def load_excel_data(folder_path: str, specific_filename: str) -> pd.DataFrame:
    file_path = os.path.join(folder_path, specific_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file được chỉ định: {file_path}")
    print(f"Đang đọc file: {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    df.dropna(subset=["Query", "Positive"], how='any', inplace=True)
    print(f"Đã tải {len(df)} hàng hợp lệ.")
    return df

def parse_references(text: str) -> Set[Tuple[str, str, str]]:
    # (Giữ nguyên hàm này)
    references = set()
    parts = re.split(r'\s+(?:và|hoặc)\s+', str(text).strip(), flags=re.IGNORECASE)
    for part in parts:
        part = part.strip().rstrip(',.')
        if not part: continue
        dieu_match = re.search(r"Điều\s+(\w+)", part, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\w+)", part, re.IGNORECASE)
        diem_match = re.search(r"Điểm\s+([a-zA-ZđĐ]+)", part, re.IGNORECASE)
        dieu = dieu_match.group(1) if dieu_match else None
        khoan = khoan_match.group(1) if khoan_match else None
        diem = diem_match.group(1) if diem_match else None
        if dieu or khoan or diem:
            dieu_str = str(dieu).lower() if dieu else None
            khoan_str = str(khoan).lower() if khoan else None
            diem_str = str(diem).lower() if diem else None
            references.add((dieu_str, khoan_str, diem_str))
    return references

def normalize_payload_ref(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    # (Giữ nguyên hàm này)
    if not isinstance(payload, dict): return (None, None, None)
    meta = payload.get("metadata", payload)
    d = meta.get("article_no")
    k = meta.get("clause_no")
    p = meta.get("point_letter") or meta.get("point_id") or None
    d_str = str(d).lower() if d is not None else None
    k_str = str(k).lower() if k is not None else None
    p_str = str(p).lower() if p is not None else None
    return (d_str, k_str, p_str)

def ground_truth_matches(retrieved_ref, gt_ref) -> bool:
    # (Giữ nguyên hàm này)
    rd, rk, rp = retrieved_ref
    gd, gk, gp = gt_ref
    if rd is None and rk is None and rp is None: return False
    if gd is not None and gd != rd: return False
    if gk is not None and gk != rk: return False
    if gp is not None and gp != rp: return False
    return True

# --- HÀM TEST CHÍNH (ĐÃ VIẾT LẠI HOÀN TOÀN ĐỂ DÙNG PREFETCH) ---
def run_single_test_case(
    client: QdrantClient,
    bge_vector: List[float],
    colbert_multi_vector: List[List[float]],
    bm25_query: str,
    ground_truth_refs: Set[Tuple[str, str, str]],
    k_dense: int,
    k_sparse: int,
    # k_candidates_from_rrf: int, # KHÔNG CẦN NỮA
    final_k: int
) -> bool:
    """
    Chạy pipeline MỚI sử dụng PREFETCH:
    Prefetch: (BGE + BM25) -> Main Query: (Colbert Rerank)
    Tất cả được thực thi trên Server Qdrant trong 1 lệnh gọi.
    """
    
    try:
        # --- BƯỚC 1: ĐỊNH NGHĨA CÁC PREFETCH QUERIES ---
        
        # Prefetch 1: BGE (Dense)
        # Đây là truy vấn để "đề cử" ứng viên
        prefetch_bge = models.Prefetch(
            query=bge_vector,
            using=BGE_VECTOR_NAME,
            limit=k_dense
        )
        
        # Prefetch 2: BM25 (Sparse)
        # Đây cũng là truy vấn "đề cử" ứng viên
        prefetch_bm25 = models.Prefetch(
            query=models.Document(text=bm25_query, model=BM25_SPARSE_MODEL_NAME),
            using=BM25_VECTOR_NAME,
            limit=k_sparse
        )
        
        # Qdrant sẽ tự động lấy TẬP HỢP (UNION) 
        # của kết quả từ prefetch_bge và prefetch_bm25

        # --- BƯỚC 2: XÂY DỰNG QUERY REQUEST HOÀN CHỈNH ---
        
        # Query chính (ColBERT) sẽ hoạt động như một RERANKER.
        # Nó chỉ chạy trên các điểm được trả về bởi các prefetch.
        
        # query_request = models.QueryRequest(
        #     # Danh sách các truy vấn đề cử (candidate nomination)
        #     prefetch=[prefetch_bge, prefetch_bm25],
            
        #     # Query chính (Reranker)
        #     query=colbert_multi_vector,
        #     using=COLBERT_VECTOR_NAME, 
            
        #     # Lấy K cuối cùng
        #     limit=final_k,
            
        #     # Cần payload để kiểm tra ground truth
        #     with_payload=True,
        # )

        # --- BƯỚC 3: CHẠY QUERY DUY NHẤT ---
        # Đây là API mới, thay thế cho .search()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            
            # --- Các tham số "bung" ra từ QueryRequest cũ ---
            prefetch=[prefetch_bge, prefetch_bm25],
            query=colbert_multi_vector,
            using=COLBERT_VECTOR_NAME, 
            limit=final_k,
            with_payload=True,
            # ---------------------------------------------
            
            with_vectors=False # Tham số này đã đúng vị trí
        )
        
        # Kết quả trả về đã được sắp xếp theo điểm ColBERT (query chính)
        top_final_k_results = results.points
        
        # --- BƯỚC 4: KIỂM TRA GROUND TRUTH ---
        is_hit_at_k = False
        for hit in top_final_k_results:
            retrieved_ref = normalize_payload_ref(hit.payload)
            for gt in ground_truth_refs:
                if ground_truth_matches(retrieved_ref, gt):
                    is_hit_at_k = True
                    break
            if is_hit_at_k:
                break
                
        return is_hit_at_k

    except Exception as e:
        print(f"Lỗi khi chạy test case (k_dense={k_dense}, k_sparse={k_sparse}) với PREFETCH: {e}")
        # In chi tiết lỗi để debug
        import traceback
        traceback.print_exc()
        return False

# --- HÀM CHÍNH (ĐÃ CẬP NHẬT) ---

def main():
    """Hàm thực thi chính của quy trình Grid Search."""
    try:
        bge_model, colbert_model, client = initialize_clients()
        df = load_excel_data(DATA_FOLDER, TEST_DATA_FILE)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi khởi tạo hoặc tải dữ liệu: {e}")
        return

    total_queries = len(df)
    param_grid = list(product(K_DENSE_VALUES, K_SPARSE_VALUES))
    
    results_matrix = {}
    for (k_dense, k_sparse) in param_grid:
        key = f"dense_{k_dense}_sparse_{k_sparse}"
        results_matrix[key] = {
            f"hit_at_{FINAL_K_AFTER_RERANK}": 0
        }
    
    # Cập nhật log
    print(f"\n--- BẮT ĐẦU GRID SEARCH (PIPELINE PREFETCH QDRANT) ---")
    print(f"Các tham số K_DENSE (BGE Prefetch): {K_DENSE_VALUES}")
    print(f"Các tham số K_SPARSE (BM25 Prefetch): {K_SPARSE_VALUES}")
    print(f"Query chính (Reranker): {COLBERT_VECTOR_NAME}")
    print(f"Đánh giá tại K cuối cùng (Sau Rerank): {FINAL_K_AFTER_RERANK}")

    first_run = True

    for index, row in df.iterrows():
        query = row["Query"]
        positive_text = row["Positive"]
        
        try:
            ground_truth_refs = parse_references(positive_text)
            if not ground_truth_refs:
                continue

            # Encode query MỘT LẦN
            bge_vector = bge_model.encode(query, convert_to_tensor=False).tolist()
            bm25_query = query
            colbert_multi_vector_np = next(colbert_model.query_embed(query))
            colbert_multi_vector = colbert_multi_vector_np.tolist()

            if first_run:
                print("\n--- DEBUG (Lần chạy đầu tiên) ---")
                print(f"Kích thước BGE vector: {len(bge_vector)}")
                num_tokens = len(colbert_multi_vector)
                token_dim = len(colbert_multi_vector[0]) if num_tokens > 0 else 0
                print(f"Kích thước Colbert multi-vector: {num_tokens} tokens x {token_dim} dim")
                print("----------------------------------\n")
                first_run = False

            # Lặp qua LƯỚI THAM SỐ
            for (k_dense, k_sparse) in param_grid:
                
                # THAY ĐỔI: Chạy logic test case mới
                is_hit = run_single_test_case(
                    client,
                    bge_vector,
                    colbert_multi_vector,
                    bm25_query,
                    ground_truth_refs,
                    k_dense,                   # Input cho BGE Prefetch
                    k_sparse,                  # Input cho BM25 Prefetch
                    FINAL_K_AFTER_RERANK       # Output cuối cùng
                )
                
                if is_hit:
                    key = f"dense_{k_dense}_sparse_{k_sparse}"
                    results_matrix[key][f"hit_at_{FINAL_K_AFTER_RERANK}"] += 1
            
            if ((index + 1) % 10) == 0:
                print(f"Đã xử lý {index + 1}/{total_queries} queries...")

        except Exception as e:
            print(f"Lỗi nghiêm trọng khi xử lý query: '{query}'. Lỗi: {e}")

    print("--- HOÀN TẤT QUÁ TRÌNH TEST ---")
    
    summary_report = {
        "test_config": {
            "test_file": TEST_DATA_FILE,
            "total_queries_processed": total_queries,
            "k_dense_values": K_DENSE_VALUES,
            "k_sparse_values": K_SPARSE_VALUES,
            # "k_candidates_from_rrf": K_CANDIDATES_FROM_RRF, # Không còn
            "final_k_after_rerank": FINAL_K_AFTER_RERANK,
            "bge_model": BGE_MODEL_NAME,
            "colbert_model": COLBERT_MODEL_NAME,
            # Cập nhật pipeline
            "pipeline": "Prefetch(BGE+BM25) -> Server-Colbert_Rerank"
        },
        "results_grid": {}
    }
    
    print(f"\n--- TÓM TẮT KẾT QUẢ: Recall@{FINAL_K_AFTER_RERANK} (%) ---")
    
    sorted_keys = sorted(results_matrix.keys(), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3])))
    
    best_recall = -1
    best_key = ""
    
    for key in sorted_keys:
        hit_count = results_matrix[key][f"hit_at_{FINAL_K_AFTER_RERANK}"]
        recall_percent = (hit_count / total_queries) * 100 if total_queries > 0 else 0
        
        summary_report["results_grid"][key] = {
            "hit_count": hit_count,
            f"recall_at_{FINAL_K_AFTER_RERANK}_percent": round(recall_percent, 2)
        }
        
        if recall_percent > best_recall:
            best_recall = recall_percent
            best_key = key
        
        print(f"{key}: {round(recall_percent, 2)}%")
    
    print(f"\n--- CẶP TỐI ƯU NHẤT ---")
    print(f"Cấu hình: {best_key}")
    print(f"Recall@{FINAL_K_AFTER_RERANK} cao nhất: {round(best_recall, 2)}%")
    summary_report["best_configuration"] = {
        "key": best_key,
        f"recall_at_{FINAL_K_AFTER_RERANK}_percent": round(best_recall, 2)
    }

    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=4)
        print(f"\nBáo cáo đã được lưu thành công tại: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"LỖI: Không thể ghi file báo cáo {OUTPUT_FILE}. Lỗi: {e}")

if __name__ == "__main__":
    main()