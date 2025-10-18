import os
import glob
import json
import re
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
TEST_DATA_FILE = "HNGD_Test.xlsx"
OUTPUT_FILE = "results/results_BAAI_retrieval.json"
TOP_K_VALUES = [5, 10, 15, 20]
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
) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]]]:
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
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=MAX_K,
        with_payload=True
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

        return {
            "id": meta.get('id') or payload.get('id'),
            "article_no": str(art) if art is not None else None,
            "clause_no": str(cla) if cla is not None else None,
            "point_letter": pt,
            "article_title": meta.get('article_title'),
            "exact_citation": meta.get('exact_citation')
        }

    retrieved_payloads = [compact_hit(hit) for hit in search_results[:MAX_K]]
    
    # 4. So sánh
    hits_at_k = {}
    found_in_max_k = False
    
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

    for k in TOP_K_VALUES:
        # Lấy K kết quả đầu tiên
        top_k_results = search_results[:k]
        
        # Chuyển đổi payload của K kết quả đó thành set các tham chiếu
        retrieved_refs_at_k = [normalize_payload_ref(hit.payload) for hit in top_k_results]
        
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
            
    return hits_at_k, found_in_max_k, retrieved_payloads

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
            hits_at_k, found_in_max_k, retrieved_payloads = run_test(
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