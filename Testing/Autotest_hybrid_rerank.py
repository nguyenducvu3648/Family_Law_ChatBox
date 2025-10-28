import os
import glob
import json
import re
import statistics
import pandas as pd
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
# THAY ĐỔI: Import thư viện fastembed cho ColBERT
from fastembed import LateInteractionTextEmbedding
from dotenv import load_dotenv
from typing import Set, Tuple, Any, Dict, List, Optional
from itertools import product # Dùng để tạo lưới tham số

# --- Cấu hình và Hằng số ---

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Lấy cấu hình từ biến môi trường
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")

# --- Cấu hình mô hình và vector (Theo tiêu chuẩn vàng) ---

# 1. Mô hình BGE (Dense Retrieval)
BGE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BGE_VECTOR_NAME = "bge-m3" # Tên index vector Dense

# 2. Mô hình Colbert (Dùng làm Reranker - Multi-Vector)
COLBERT_MODEL_NAME = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
COLBERT_VECTOR_NAME = "colbertv2.0" # TÊN INDEX MULTI-VECTOR (128-dim)

# 3. Vector BM25 (Sparse Retrieval)
BM25_VECTOR_NAME = "bm25" # Tên index vector Sparse
BM25_SPARSE_MODEL_NAME = "Qdrant/bm25"
# ----------------------------------------------

# Tự động chọn device (GPU nếu có)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cấu hình LƯỚI TEST (GRID SEARCH) ---
DATA_FOLDER = "data"
TEST_DATA_FILE = "HNGD_Full.xlsx"
OUTPUT_FILE = "results/results_HNGD_V1.json"

# Các giá trị K cho từng tầng retrieval
K_DENSE_VALUES = [10, 20, 30, 50, 70, 100]
K_SPARSE_VALUES = [10, 20, 30, 50, 70, 100]

# Số lượng kết quả cuối cùng SAU KHI RERANK
FINAL_K_AFTER_RERANK = 7
# ----------------------------------------------

# --- Các Hàm Tiện Ích ---

# THAY ĐỔI: Sửa hàm khởi tạo
def initialize_clients() -> Tuple[SentenceTransformer, LateInteractionTextEmbedding, QdrantClient]:
    """
    Khởi tạo mô hình BGE (SentenceTransformer) và
    Colbert (FastEmbed) và Qdrant client.
    """
    print(f"Sử dụng device: {DEVICE}")
    
    # 1. Tải mô hình BGE (cho Giai đoạn 1: Retrieve)
    print(f"Đang tải mô hình BGE: {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
    
    # 2. Tải mô hình Colbert (Bằng thư viện FastEmbed)
    print(f"Đang tải mô hình Colbert (FastEmbed): {COLBERT_MODEL_NAME}...")
    # FastEmbed's wrapper sẽ tự động xử lý tokenization, 
    # projection 768->128, chuẩn hóa L2 và trả về multi-vector
    colbert_model = LateInteractionTextEmbedding(
        model_name=COLBERT_MODEL_NAME, 
        device=DEVICE
    )
    
    # 3. Kết nối Qdrant
    print(f"Đang kết nối tới Qdrant tại: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Kết nối Qdrant và collection '{COLLECTION_NAME}' thành công.")
    except Exception as e:
        print(f"LỖI: Không thể kết nối hoặc tìm thấy collection '{COLLECTION_NAME}'.")
        print(f"Chi tiết lỗi: {e}")
        raise
        
    # THAY ĐỔI: Trả về colbert_model của FastEmbed
    return bge_model, colbert_model, client

def load_excel_data(folder_path: str, specific_filename: str) -> pd.DataFrame:
    """Tải và xác thực file Excel"""
    
    if not specific_filename:
        raise ValueError("Chưa chỉ định file test data.")
        
    file_path = os.path.join(folder_path, specific_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file được chỉ định: {file_path}")

    print(f"Đang đọc file: {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    
    if "Query" not in df.columns or "Positive" not in df.columns:
        raise ValueError("File Excel phải chứa 2 cột bắt buộc: 'Query' và 'Positive'.")
        
    df.dropna(subset=["Query", "Positive"], how='any', inplace=True)
    print(f"Đã tải {len(df)} hàng hợp lệ.")
    return df

def parse_references(text: str) -> Set[Tuple[str, str, str]]:
    """Trích xuất tham chiếu (Ground Truth)"""
    references = set()
    parts = re.split(r'\s+(?:và|hoặc)\s+', str(text).strip(), flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip().rstrip(',.')
        if not part: continue
            
        dieu_match = re.search(r"Điều\s+(\w+)", part, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\w+)", part, re.IGNORECASE)
        diem_match = re.search(r"Điểm\s+([a-zA-ZđĐ]+)", part, re.IGNORECASE) # Giả định "Điểm" là chữ
        
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
    """Chuẩn hóa payload (kết quả retrieve)"""
    if not isinstance(payload, dict):
        return (None, None, None)

    meta = payload.get("metadata", payload)

    d = meta.get("article_no")
    k = meta.get("clause_no")
    p = meta.get("point_letter") or meta.get("point_id") or None

    d_str = str(d).lower() if d is not None else None
    k_str = str(k).lower() if k is not None else None
    p_str = str(p).lower() if p is not None else None

    return (d_str, k_str, p_str)

def run_single_test_case(
    client: QdrantClient,
    bge_vector: List[float],
    colbert_multi_vector: List[List[float]], # Đây là colbertv2.0 (multi-vector)
    bm25_query: str,
    ground_truth_refs: Set[Tuple[str, str, str]],
    k_dense: int,
    k_sparse: int,
    final_k: int
) -> bool:
    """
    Chạy pipeline: (BGE + BM25) Retrieve -> (Colbert-MultiVector) Rerank
    """
    
    try:
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            
            # Giai đoạn 1: Lấy ứng viên (BGE + BM25)
            prefetch=[
                models.Prefetch(
                    query=bge_vector, # Dùng BGE-Dense
                    using=BGE_VECTOR_NAME,
                    limit=k_dense
                ),
                models.Prefetch(
                    query=models.Document(
                        text=bm25_query,
                        model=BM25_SPARSE_MODEL_NAME 
                    ), 
                    using=BM25_VECTOR_NAME,
                    limit=k_sparse
                )
            ],
            
            # Giai đoạn 2: Rerank bằng multi-vector Colbert
            query=colbert_multi_vector,
            
            # Chỉ định Qdrant dùng index 'colbertv2.0' (Multi-Vector)
            using=COLBERT_VECTOR_NAME, 
            
            limit=final_k, # Lấy top 7
            with_payload=True
        )
        
        # SỬA LỖI: Phải truy cập thuộc tính .points của kết quả trả về
        top_final_k_results = search_results.points
        
        def ground_truth_matches(retrieved_ref, gt_ref) -> bool:
            rd, rk, rp = retrieved_ref
            gd, gk, gp = gt_ref
            
            if rd is None and rk is None and rp is None:
                return False
                
            if gd is not None and gd != rd: return False
            if gk is not None and gk != rk: return False
            if gp is not None and gp != rp: return False
            return True
            
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
        print(f"Lỗi khi chạy test case (k_dense={k_dense}, k_sparse={k_sparse}): {e}")
        return False
# --- Hàm Chính (Thực thi Grid Search) ---

def main():
    """Hàm thực thi chính của quy trình Grid Search."""
    try:
        # THAY ĐỔI: Nhận về model FastEmbed cho ColBERT
        bge_model, colbert_model, client = initialize_clients()
        df = load_excel_data(DATA_FOLDER, TEST_DATA_FILE)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi khởi tạo hoặc tải dữ liệu: {e}")
        return

    total_queries = len(df)
    
    # 1. Tạo lưới tham số (các cặp k)
    param_grid = list(product(K_DENSE_VALUES, K_SPARSE_VALUES))
    
    # 2. Khởi tạo cấu trúc báo cáo (ma trận kết quả)
    results_matrix = {}
    for (k_dense, k_sparse) in param_grid:
        key = f"dense_{k_dense}_sparse_{k_sparse}"
        results_matrix[key] = {
            f"hit_at_{FINAL_K_AFTER_RERANK}": 0
        }
    
    print(f"\n--- BẮT ĐẦU GRID SEARCH ({total_queries} QUERIES x {len(param_grid)} CẶP THAM SỐ) ---")
    print(f"Các tham số K_DENSE (BGE): {K_DENSE_VALUES}")
    print(f"Các tham số K_SPARSE (BM25): {K_SPARSE_VALUES}")
    print(f"Đánh giá tại K cuối cùng (Sau Rerank): {FINAL_K_AFTER_RERANK}")

    # Biến kiểm tra
    first_run = True

    # Lặp qua từng HÀNG (query) trong DataFrame
    for index, row in df.iterrows():
        query = row["Query"]
        positive_text = row["Positive"]
        
        try:
            # 1. Lấy ground truth
            ground_truth_refs = parse_references(positive_text)
            if not ground_truth_refs:
                print(f"Cảnh báo: Không thể trích xuất tham chiếu cho query: '{query}'. Bỏ qua.")
                continue

            # 2. Encode query MỘT LẦN (tiết kiệm thời gian)
            bge_vector = bge_model.encode(query, convert_to_tensor=False).tolist()
            bm25_query = query # BM25 dùng query text

            # --- SỬA LỖI: Encode ColBERT bằng FastEmbed ---
            # .query_embed() trả về 1 iterator. next() lấy phần tử đầu tiên (query)
            # Kết quả là 1 numpy array (num_tokens, 128)
            colbert_multi_vector_np = next(colbert_model.query_embed(query))
            # Chuyển sang list[list[float]] mà Qdrant client cần
            colbert_multi_vector = colbert_multi_vector_np.tolist()
            # --------------------------------------------------

            # In debug cho lần chạy đầu tiên
            if first_run:
                print("\n--- DEBUG (Lần chạy đầu tiên) ---")
                print(f"Kích thước BGE vector: {len(bge_vector)}")
                # THAY ĐỔI: Debug cho multi-vector
                num_tokens = len(colbert_multi_vector)
                token_dim = len(colbert_multi_vector[0]) if num_tokens > 0 else 0
                print(f"Kích thước Colbert multi-vector: {num_tokens} tokens x {token_dim} dim")
                
                if token_dim != 128:
                    print(f"CẢNH BÁO: Vector ColBERT có dim={token_dim}, nhưng index của bạn là 128!")
                print("----------------------------------\n")
                first_run = False


            # 3. Lặp qua LƯỚI THAM SỐ
            for (k_dense, k_sparse) in param_grid:
                
                # Chạy test case cho cặp (k_dense, k_sparse) này
                is_hit = run_single_test_case(
                    client,
                    bge_vector,
                    colbert_multi_vector, # THAY ĐỔI: Truyền multi-vector
                    bm25_query,
                    ground_truth_refs,
                    k_dense,
                    k_sparse,
                    FINAL_K_AFTER_RERANK
                )
                
                # 4. Ghi lại kết quả
                if is_hit:
                    key = f"dense_{k_dense}_sparse_{k_sparse}"
                    results_matrix[key][f"hit_at_{FINAL_K_AFTER_RERANK}"] += 1
            
            # In tiến độ
            if ((index + 1) % 10) == 0:
                print(f"Đã xử lý {index + 1}/{total_queries} queries...")

        except Exception as e:
            print(f"Lỗi nghiêm trọng khi xử lý query: '{query}'. Lỗi: {e}")

    print("--- HOÀN TẤT QUÁ TRÌNH TEST ---")
    
    # 5. Tính toán kết quả cuối cùng (tỷ lệ recall)
    summary_report = {
        "test_config": {
            "test_file": TEST_DATA_FILE,
            "total_queries_processed": total_queries,
            "k_dense_values": K_DENSE_VALUES,
            "k_sparse_values": K_SPARSE_VALUES,
            "final_k_after_rerank": FINAL_K_AFTER_RERANK,
            "bge_model": BGE_MODEL_NAME,
            "colbert_model": COLBERT_MODEL_NAME,
            # Cập nhật pipeline
            "pipeline": "BGE_Retrieve + BM25_Retrieve -> Colbert(Multi-Vector)_Rerank"
        },
        "results_grid": {}
    }
    
    print(f"\n--- TÓM TẮT KẾT QUẢ: Recall@{FINAL_K_AFTER_RERANK} (%) ---")
    
    # Sắp xếp để dễ đọc
    sorted_keys = sorted(results_matrix.keys(), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3])))
    
    best_recall = -1
    best_key = ""
    
    for key in sorted_keys:
        hit_count = results_matrix[key][f"hit_at_{FINAL_K_AFTER_RERANK}"]
        recall_percent = (hit_count / total_queries) * 100 if total_queries > 0 else 0
        
        # Thêm vào báo cáo
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

    # 6. Lưu file JSON
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=4)
        print(f"\nBáo cáo đã được lưu thành công tại: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"LỖI: Không thể ghi file báo cáo {OUTPUT_FILE}. Lỗi: {e}")

if __name__ == "__main__":
    main()