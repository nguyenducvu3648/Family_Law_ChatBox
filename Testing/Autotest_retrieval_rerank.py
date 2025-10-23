import os
import glob
import json
import re
import statistics
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Set, Tuple, Any, Dict, List

### THAY ĐỔI ###
# Thêm CrossEncoder để rerank
from sentence_transformers import CrossEncoder

# --- Cấu hình và Hằng số ---

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Lấy cấu hình từ biến môi trường
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

### THAY ĐỔI ###
# Thêm mô hình Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# QUAN TRỌNG: Đặt tên trường payload chứa text của chunk
# Đây là trường mà reranker sẽ đọc. Ví dụ: 'text_chunk', 'content', 'text', 'van_ban'
TEXT_PAYLOAD_FIELD = "text_chunk" 

# Các hằng số cho bài test
DATA_FOLDER = "data"
TEST_DATA_FILE = "HNGD_Test.xlsx"
### THAY ĐỔI ###
# Đổi tên file output để phân biệt
OUTPUT_FILE = "results/results_BAAI_HNGD_retrieval_RERANKED_V1.json"
TOP_K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85 ,90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150 ]
MAX_K = max(TOP_K_VALUES) # Đây là số lượng retrieve ban đầu (K-retrieval)

# --- Các Hàm Tiện Ích ---

### THAY ĐỔI ###
def initialize_clients() -> Tuple[SentenceTransformer, QdrantClient, CrossEncoder]:
    """Khởi tạo mô hình embedding, Qdrant client và Reranker."""
    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Đang tải mô hình reranker: {RERANKER_MODEL}...")
    reranker = CrossEncoder(RERANKER_MODEL)
    
    print(f"Đang kết nối tới Qdrant tại: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    # Kiểm tra kết nối và collection
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Kết nối Qdrant và collection '{COLLECTION_NAME}' thành công.")
    except Exception as e:
        print(f"LỖI: Không thể kết nối hoặc tìm thấy collection '{COLLECTION_NAME}'.")
        print(f"Chi tiết lỗi: {e}")
        raise
        
    return model, client, reranker

def load_excel_data(folder_path: str, specific_filename: str) -> pd.DataFrame:
    """Tải và xác thực file Excel được chỉ định từ thư mục."""
    
    if not specific_filename:
        print(f"LỖI: Biến môi trường 'TEST_DATA_FILE' chưa được_set trong file .env.")
        print(f"Vui lòng thêm TEST_DATA_FILE=ten_file_cua_ban.xlsx vào file .env.")
        raise ValueError("Chưa chỉ định file test data.")
        
    file_path = os.path.join(folder_path, specific_filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file được chỉ định: {file_path}")

    print(f"Đang đọc file: {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 1. Kiểm tra header
    if "Query" not in df.columns or "Positive" not in df.columns:
        raise ValueError("File Excel phải chứa 2 cột bắt buộc: 'Query' và 'Positive'.")
        
    # 2. Loại bỏ các hàng trống (nơi cả Query và Positive đều là NaN)
    initial_rows = len(df)
    df.dropna(subset=["Query", "Positive"], how='all', inplace=True)
    # Loại bỏ các hàng mà 1 trong 2 cột thiết yếu bị thiếu
    df.dropna(subset=["Query", "Positive"], how='any', inplace=True)
    final_rows = len(df)
    
    print(f"Đã tải {final_rows} hàng hợp lệ (loại bỏ {initial_rows - final_rows} hàng trống/không hợp lệ).")
    return df

def parse_references(text: str) -> Set[Tuple[str, str, str]]:
    """
    Trích xuất các tham chiếu (Điều, Khoản, Điểm) từ văn bản "Positive".
    Ví dụ: "Điều 5, Khoản 1, Điểm a và Điều 8, Khoản 2"
    Kết quả: {('5', '1', 'a'), ('8', '2', None)}
    """
    references = set()
    
    # Tách các cụm tham chiếu dựa trên "và" hoặc "hoặc"
    parts = re.split(r'\s+(?:và|hoặc)\s+', str(text).strip(), flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip().rstrip(',.')
        if not part:
            continue
            
        # Sử dụng \w+ để bắt được "5", "1", "a"
        dieu_match = re.search(r"Điều\s+(\w+)", part, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\w+)", part, re.IGNORECASE)
        diem_match = re.search(r"Điểm\s+(\w+)", part, re.IGNORECASE)
        
        dieu = dieu_match.group(1) if dieu_match else None
        khoan = khoan_match.group(1) if khoan_match else None
        diem = diem_match.group(1) if diem_match else None
        
        # Chỉ thêm nếu có ít nhất một thông tin
        if dieu or khoan or diem:
            references.add((dieu, khoan, diem))
            
    return references

def normalize_payload_ref(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Chuẩn hóa payload từ Qdrant thành một tuple (Điều, Khoản, Điểm).
    Sử dụng key từ metadata: 'article_no', 'clause_no', 'point_letter'.
    """
    # Payload may be stored directly (article_no at top-level) or nested under a 'metadata' key
    if not isinstance(payload, dict):
        return (None, None, None)

    meta = None
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        meta = payload["metadata"]
    else:
        # payload may already be the metadata dict
        meta = payload

    d = meta.get("article_no")
    k = meta.get("clause_no")
    p = meta.get("point_letter") or meta.get("point_id") or None

    # Chuẩn hóa: convert số nguyên -> string, giữ None nếu không tồn tại
    d_str = str(d) if d is not None else None
    k_str = str(k) if k is not None else None

    # point letter normalize: lower-case single letters; if it's like 'a' or 'A' keep it
    if p is None:
        p_str = None
    else:
        p_str = str(p).strip()
        # If it's a single alphabetic char followed by ')' or similar, strip non-alpha
        p_str = re.sub(r"[^a-zA-ZđĐ]", "", p_str)
        p_str = p_str.lower() if p_str else None

    return (d_str, k_str, p_str)

### THAY ĐỔI ###
def run_test(
    query: str, 
    ground_truth_refs: Set[Tuple[str, str, str]], 
    model: SentenceTransformer, 
    client: QdrantClient,
    reranker: CrossEncoder  # Thêm reranker
) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]], Any]:
    """
    Thực hiện embedding, retrieval, RERANKING và so sánh cho một query.
    Trả về:
    1. Dict kết quả hit/miss cho từng mốc K.
    2. Bool tổng quát: có tìm thấy ở K cao nhất không.
    3. List các payload đã retrieve (ĐÃ RERANK, để báo cáo lỗi).
    """
    
    # 1. Embedding
    query_vector = model.encode(query, convert_to_tensor=False).tolist()
    
    # 2. Retrieval (Lấy K lớn nhất, ví dụ 150)
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=MAX_K,
        with_payload=True,
        with_vectors=False # Không cần trả về vector
    )
    
    top_max_results = search_results.points # Đây là list các ScoredPoint
    
    # --- BƯỚC 2.5: RERANKING ---
    # print(f"DEBUG: Bắt đầu rerank {len(top_max_results)} kết quả...")
    
    # 2.5.1. Chuẩn bị các cặp [query, text]
    # QUAN TRỌNG: Đảm bảo payload của bạn chứa trường văn bản thực tế.
    # Thay 'text_chunk' bằng tên trường chính xác trong Qdrant payload của bạn (ví dụ: 'content', 'text', 'van_ban', v.v.)
    rerank_pairs = []
    hits_without_text = [] # Để theo dõi các hit không có text
    
    for i, hit in enumerate(top_max_results):
        doc_text = None
        if isinstance(hit.payload, dict):
            # Cố gắng lấy text từ payload gốc
            doc_text = hit.payload.get(TEXT_PAYLOAD_FIELD)
            
            # Fallback nếu text nằm trong 'metadata'
            if not doc_text and isinstance(hit.payload.get('metadata'), dict):
                 doc_text = hit.payload.get('metadata', {}).get(TEXT_PAYLOAD_FIELD)

        if doc_text and isinstance(doc_text, str):
            rerank_pairs.append([query, doc_text])
        else:
            # Nếu không tìm thấy text, đánh dấu để xử lý
            # print(f"CẢNH BÁO: Không tìm thấy trường text '{TEXT_PAYLOAD_FIELD}' trong payload cho ID: {hit.id}. Sẽ gán score rerank = -inf.")
            hits_without_text.append(i) # Lưu lại chỉ số (index)
            rerank_pairs.append(None) # Thêm placeholder

    # 2.5.2. Tính điểm rerank
    rerank_scores = []
    valid_pairs = [pair for pair in rerank_pairs if pair is not None]
    
    if valid_pairs:
        # Chỉ đưa các cặp hợp lệ vào model
        scores_from_model = reranker.predict(valid_pairs, convert_to_tensor=False).tolist()
        
        # Map điểm về lại list ban đầu
        score_idx = 0
        for pair in rerank_pairs:
            if pair is not None:
                rerank_scores.append(scores_from_model[score_idx])
                score_idx += 1
            else:
                rerank_scores.append(float('-inf')) # Đẩy các kết quả không có text xuống cuối
    else:
        # Trường hợp không có valid pairs nào
        rerank_scores = [float('-inf')] * len(rerank_pairs)

    # 2.5.3. Kết hợp kết quả và sắp xếp lại
    # Gói (hit_object, original_retrieval_score, new_rerank_score)
    combined_results = []
    for hit, rerank_score in zip(top_max_results, rerank_scores):
        combined_results.append({
            "hit_object": hit, # ScoredPoint object (chứa payload, id, score cũ)
            "retrieval_score": hit.score, 
            "rerank_score": rerank_score
        })

    # Sắp xếp lại dựa trên rerank_score (cao nhất -> thấp nhất)
    combined_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # List kết quả cuối cùng (chỉ chứa các ScoredPoint) đã được sắp xếp lại
    reranked_results = [item["hit_object"] for item in combined_results]
    
    # print("DEBUG: Rerank hoàn tất.")
    # --- KẾT THÚC RERANKING ---


    # 3. Chuẩn hóa kết quả retrieve (ĐÃ RERANK)
    def compact_hit(hit) -> Dict[str, Any]:
        payload = hit.payload if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
        meta = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else payload

        art = meta.get('article_no')
        cla = meta.get('clause_no')
        pt = meta.get('point_letter') or meta.get('point_id') or None
        if pt is not None:
            pt = re.sub(r"[^a-zA-ZđĐ]", "", str(pt)).lower() or None

        # Lấy retrieval score (từ Qdrant)
        retrieval_score = None
        try:
            retrieval_score = getattr(hit, 'score', None)
        except Exception:
            retrieval_score = None
            
        # Lấy rerank score (từ combined_results, cần tìm lại)
        # Cách đơn giản: Tìm trong combined_results
        # Lưu ý: 'hit' ở đây là 'hit_object'
        rerank_score_found = None
        for item in combined_results:
            if item["hit_object"].id == hit.id:
                 rerank_score_found = item["rerank_score"]
                 break

        return {
            "id": meta.get('id') or payload.get('id'),
            "article_no": str(art) if art is not None else None,
            "clause_no": str(cla) if cla is not None else None,
            "point_letter": pt,
            "article_title": meta.get('article_title'),
            "exact_citation": meta.get('exact_citation'),
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score_found # Thêm điểm rerank vào log
        }

    # Lấy payload đã được SẮP XẾP LẠI (reranked)
    retrieved_payloads = [compact_hit(hit) for hit in reranked_results]
    
    # 4. So sánh
    hits_at_k = {}
    found_in_max_k = False
    first_hit_rank = None
    
    def ground_truth_matches(retrieved_ref: Tuple[str,str,str], gt_ref: Tuple[str,str,str]) -> bool:
        """Return True if retrieved_ref matches gt_ref where gt None acts as wildcard."""
        rd, rk, rp = retrieved_ref
        gd, gk, gp = gt_ref
        # Compare article
        if gd is not None and rd is not None:
            if gd != rd:
                return False
        elif gd is not None and rd is None:
            return False
        # Compare clause (None in ground truth means wildcard)
        if gk is not None:
            if rk is None:
                return False
            if gk != rk:
                return False
        # Compare point
        if gp is not None:
            if rp is None:
                return False
            if gp != rp:
                return False
        return True

    # Precompute normalized refs cho KẾT QUẢ ĐÃ RERANK
    retrieved_refs_all = [normalize_payload_ref(hit.payload) for hit in reranked_results]

    # Lấy RERANK scores cho top_max_results
    retrieved_scores_all = [item["rerank_score"] for item in combined_results]

    # determine first hit rank (1-based) and its score if any
    first_hit_score = None
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

    for k in TOP_K_VALUES:
        # Lấy K kết quả đầu tiên TỪ DANH SÁCH ĐÃ RERANK
        # top_k_results = reranked_results[:k] # Không cần thiết, đã có retrieved_refs_all

        # Chuyển đổi payload của K kết quả đó thành set các tham chiếu
        retrieved_refs_at_k = retrieved_refs_all[:k]
        
        # Debug: In cho query đầu tiên
        try:
            if run_test.debug_count < 1:
                print(f"DEBUG Query: {query}")
                print(f"DEBUG Ground truth: {ground_truth_refs}")
                print(f"DEBUG RERANKED Retrieved at {k}: {retrieved_refs_at_k}")
                run_test.debug_count += 1
        except AttributeError:
            run_test.debug_count = 1
            print(f"DEBUG Query: {query}")
            print(f"DEBUG Ground truth: {ground_truth_refs}")
            print(f"DEBUG RERANKED Retrieved at {k}: {retrieved_refs_at_k}")
        
        # Kiểm tra
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
            found_in_max_k = True # Đánh dấu đã tìm thấy
            
    # Trả về điểm first_hit_score (đã là rerank score)
    return hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score

# --- Hàm Chính ---

def main():
    """Hàm thực thi chính của quy trình test."""
    try:
        ### THAY ĐỔI ###
        model, client, reranker = initialize_clients()
        df = load_excel_data(DATA_FOLDER, TEST_DATA_FILE)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi khởi tạo hoặc tải dữ liệu: {e}")
        return

    total_queries = len(df)
    
    # Khởi tạo cấu trúc báo cáo
    summary = {
        "test_run_parameters": {
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "collection_name": COLLECTION_NAME,
            "retrieval_k_before_rerank": MAX_K,
            "text_payload_field_used": TEXT_PAYLOAD_FIELD
        },
        "total_queries_in_file": total_queries,
        "scanned_queries": 0,
        "queries_with_no_hit": 0,
    }
    summary.update({f"hit_at_{k}": 0 for k in TOP_K_VALUES})
    
    missed_queries_details = []
    first_hit_ranks: List[int] = []
    queries_with_any_first_hit = 0
    first_hit_scores: List[float] = []

    print(f"\n--- BẮT ĐẦU QUÁ TRÌNH TEST (với Reranker: {RERANKER_MODEL}) ---")

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

            # 2. Chạy test (với reranker)
            ### THAY ĐỔI ###
            hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score = run_test(
                query, ground_truth_refs, model, client, reranker
            )
            
            # 3. Cập nhật thống kê
            summary["scanned_queries"] += 1
            for k_str, is_hit in hits_at_k.items():
                if is_hit:
                    summary[k_str] += 1
                    
            # 4. Ghi lại các trường hợp "miss" (sau khi đã rerank)
            if not found_in_max_k:
                summary["queries_with_no_hit"] += 1
                missed_queries_details.append({
                    "query": query,
                    "expected_references": [str(ref) for ref in ground_truth_refs],
                    f"retrieved_and_reranked_top_{MAX_K}": retrieved_payloads
                })
            
            # record first hit rank stats
            if first_hit_rank is not None:
                first_hit_ranks.append(first_hit_rank)
                queries_with_any_first_hit += 1
            if first_hit_score is not None and first_hit_score != float('-inf'): # Bỏ qua các score -inf
                try:
                    first_hit_scores.append(float(first_hit_score))
                except Exception:
                    pass

            # In tiến độ
            if (summary["scanned_queries"] % 10) == 0:
                print(f"Đã xử lý {summary['scanned_queries']}/{total_queries} queries...")

        except Exception as e:
            print(f"Lỗi khi xử lý query: '{query}'. Lỗi: {e}")
            import traceback
            traceback.print_exc() # In chi tiết lỗi
            missed_queries_details.append({
                "query": query,
                "error": str(e)
            })

    print("--- HOÀN TẤT QUÁ TRÌNH TEST ---")
    
    # Tính toán recall
    for k in TOP_K_VALUES:
        hit_count = summary[f"hit_at_{k}"]
        recall = (hit_count / summary["scanned_queries"]) * 100 if summary["scanned_queries"] > 0 else 0
        summary[f"recall_at_{k}_percent"] = round(recall, 2)

    # Tính thống kê first-hit
    if first_hit_ranks:
        avg_first_hit = sum(first_hit_ranks) / len(first_hit_ranks)
    else:
        avg_first_hit = None

    summary["avg_first_hit_rank"] = round(avg_first_hit, 2) if avg_first_hit is not None else None
    summary["pct_queries_with_first_hit_within_top_{0}".format(MAX_K)] = round((queries_with_any_first_hit / summary["scanned_queries"])*100, 2) if summary["scanned_queries"]>0 else 0.0
    
    # Thống kê điểm (đây là RERANK score)
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

    score_stats = {
        "avg_first_hit_rerank_score": round(avg_first_hit_score, 4) if avg_first_hit_score is not None else None,
        "std_first_hit_rerank_score": round(std_first_hit_score, 4) if std_first_hit_score is not None else None,
        "max_first_hit_rerank_score": round(max_first_hit_score, 4) if max_first_hit_score is not None else None,
        "min_first_hit_rerank_score": round(min_first_hit_score, 4) if min_first_hit_score is not None else None
    }
    
    # Reranker BAAI thường có điểm dương cao cho các cặp tốt, và điểm âm cho các cặp xấu.
    # Ngưỡng 0.0 là một điểm khởi đầu tốt.
    score_stats["suggested_rerank_score_threshold"] = 0.0
    summary["first_hit_rerank_score_stats"] = score_stats


    # 5. Tạo báo cáo cuối cùng
    report = {
        "summary": summary,
        "missed_queries_details": missed_queries_details
    }
    
    # 6. Lưu file JSON
    try:
        # Đảm bảo thư mục results tồn tại
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print(f"\nBáo cáo đã được lưu thành công tại: {OUTPUT_FILE}")
        
        # In tóm tắt ra console
        print("\n--- TÓM TẮT KẾT QUẢ (ĐÃ RERANK) ---")
        print(json.dumps(summary, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"LỖI: Không thể ghi file báo cáo {OUTPUT_FILE}. Lỗi: {e}")

if __name__ == "__main__":
    main()