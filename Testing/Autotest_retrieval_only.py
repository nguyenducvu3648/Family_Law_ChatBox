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

# --- Cấu hình và Hằng số ---

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Lấy cấu hình từ biến môi trường
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Các hằng số cho bài test
DATA_FOLDER = "data"
TEST_DATA_FILE = "HNGD_Full.xlsx"
OUTPUT_FILE = "results/results_BAAI_HNGD_retrieval_only_V2.json"
TOP_K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85 ,90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150 ]
MAX_K = max(TOP_K_VALUES)

# --- Các Hàm Tiện Ích ---

def initialize_clients() -> Tuple[SentenceTransformer, QdrantClient]:
    """Khởi tạo mô hình embedding và Qdrant client."""
    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
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
        
    return model, client

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

def run_test(
    query: str, 
    ground_truth_refs: Set[Tuple[str, str, str]], 
    model: SentenceTransformer, 
    client: QdrantClient
) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]], Any]:
    """
    Thực hiện embedding, retrieval và so sánh cho một query.
    Trả về:
    1. Dict kết quả hit/miss cho từng mốc K.
    2. Bool tổng quát: có tìm thấy ở K cao nhất không.
    3. List các payload đã retrieve (để báo cáo lỗi).
    """
    
    # 1. Embedding
    query_vector = model.encode(query, convert_to_tensor=False).tolist()
    
    # 2. Retrieval
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=MAX_K,
        with_payload=True,
        using="bge-m3"
    )
    
    # 3. Chuẩn hóa kết quả retrieve
    # We'll keep a compact view of top MAX_K retrieved items (metadata only) to include in miss reports
    def compact_hit(hit) -> Dict[str, Any]:
        # hit is expected to have .payload (dict)
        payload = hit.payload if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
        meta = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else payload

        # Normalize values
        art = meta.get('article_no')
        cla = meta.get('clause_no')
        pt = meta.get('point_letter') or meta.get('point_id') or None
        if pt is not None:
            pt = re.sub(r"[^a-zA-ZđĐ]", "", str(pt)).lower() or None

        # try to extract score if present on hit object
        score = None
        try:
            score = getattr(hit, 'score', None)
        except Exception:
            score = None

        return {
            "id": meta.get('id') or payload.get('id'),
            "article_no": str(art) if art is not None else None,
            "clause_no": str(cla) if cla is not None else None,
            "point_letter": pt,
            "article_title": meta.get('article_title'),
            "exact_citation": meta.get('exact_citation'),
            "score": score
        }

    top_max_results = search_results.points
    retrieved_payloads = [compact_hit(hit) for hit in top_max_results]
    
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

    # Precompute normalized refs for top MAX_K results
    # top_max_results = search_results[:MAX_K]
    retrieved_refs_all = [normalize_payload_ref(hit.payload) for hit in top_max_results]

    # collect scores for top_max_results
    retrieved_scores_all = []
    for hit in top_max_results:
        sc = None
        try:
            sc = getattr(hit, 'score', None)
        except Exception:
            sc = None
        # fallback: if payload contains 'score'
        if sc is None and isinstance(hit, dict):
            sc = hit.get('payload', {}).get('score') if isinstance(hit.get('payload', {}), dict) else None
        retrieved_scores_all.append(sc)

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
        # Lấy K kết quả đầu tiên
        top_k_results = top_max_results[:k]

        # Chuyển đổi payload của K kết quả đó thành set các tham chiếu
        retrieved_refs_at_k = retrieved_refs_all[:k]
        
        # Debug: In cho query đầu tiên (sử dụng biến global hoặc flag)
        try:
            if run_test.debug_count < 1:
                print(f"DEBUG Query: {query}")
                print(f"DEBUG Ground truth: {ground_truth_refs}")
                print(f"DEBUG Retrieved at {k}: {retrieved_refs_at_k}")
                run_test.debug_count += 1
        except AttributeError:
            run_test.debug_count = 1
            print(f"DEBUG Query: {query}")
            print(f"DEBUG Ground truth: {ground_truth_refs}")
            print(f"DEBUG Retrieved at {k}: {retrieved_refs_at_k}")
        
        # Kiểm tra xem có bất kỳ tham chiếu "ground truth" nào
        # khớp (với None trong ground truth là wildcard) với các tham chiếu retrieve được
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
            
    return hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score

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
        "total_queries_in_file": total_queries,
        "scanned_queries": 0,
        "queries_with_no_hit": 0,
    }
    summary.update({f"hit_at_{k}": 0 for k in TOP_K_VALUES})
    
    missed_queries_details = []
    first_hit_ranks: List[int] = []
    queries_with_any_first_hit = 0
    first_hit_scores: List[float] = []

    print(f"\n--- BẮT ĐẦU QUÁ TRÌNH TEST ({total_queries} QUERIES) ---")

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

            # 2. Chạy test
            hits_at_k, found_in_max_k, retrieved_payloads, first_hit_rank, first_hit_score = run_test(
                query, ground_truth_refs, model, client
            )
            
            # 3. Cập nhật thống kê
            summary["scanned_queries"] += 1
            for k_str, is_hit in hits_at_k.items():
                if is_hit:
                    summary[k_str] += 1
                    
            # 4. Ghi lại các trường hợp "miss"
            if not found_in_max_k:
                summary["queries_with_no_hit"] += 1
                # Only include compact retrieved metadata for misses to keep the report focused and smaller
                missed_queries_details.append({
                    "query": query,
                    "expected_references": [str(ref) for ref in ground_truth_refs],
                    f"retrieved_top_{MAX_K}": retrieved_payloads
                })
            # record first hit rank stats
            if first_hit_rank is not None:
                first_hit_ranks.append(first_hit_rank)
                queries_with_any_first_hit += 1
            if first_hit_score is not None:
                try:
                    first_hit_scores.append(float(first_hit_score))
                except Exception:
                    pass

            # In tiến độ
            if (summary["scanned_queries"] % 10) == 0:
                print(f"Đã xử lý {summary['scanned_queries']}/{total_queries} queries...")

        except Exception as e:
            print(f"Lỗi khi xử lý query: '{query}'. Lỗi: {e}")
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
    # Score stats
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
    # simple suggested threshold: mean - stddev (users can choose more conservative value)
    if avg_first_hit_score is not None and std_first_hit_score is not None:
        suggested = avg_first_hit_score - std_first_hit_score
        summary["suggested_score_threshold_mean_minus_std"] = round(suggested, 4)
    else:
        summary["suggested_score_threshold_mean_minus_std"] = None

    # 5. Tạo báo cáo cuối cùng
    report = {
        "summary": summary,
        "missed_queries_details": missed_queries_details
    }
    
    # 6. Lưu file JSON
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print(f"\nBáo cáo đã được lưu thành công tại: {OUTPUT_FILE}")
        
        # In tóm tắt ra console
        print("\n--- TÓM TẮT KẾT QUẢ ---")
        print(json.dumps(summary, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"LỖI: Không thể ghi file báo cáo {OUTPUT_FILE}. Lỗi: {e}")

if __name__ == "__main__":
    main()