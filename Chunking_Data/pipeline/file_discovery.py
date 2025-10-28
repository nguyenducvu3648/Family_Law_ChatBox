#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Discovery Module
=====================

Module tìm kiếm và catalog các file luật và câu hỏi trong law_content.

Functions:
  - find_law_files(): Tìm tất cả file .doc/.docx
  - find_question_files(): Tìm tất cả file câu hỏi .xlsx
  - create_file_paths_list(): Tạo danh sách file paths
"""

import os
import glob
from typing import List, Dict, Any, Tuple


def find_law_files(
    law_content_dir: str = "law_content",
    verbose: bool = True
) -> Tuple[List[str], Dict[str, Dict[str, List[Dict]]]]:
    """
    Tìm tất cả file văn bản pháp lý trong thư mục law_content.
    
    Chỉ tìm files trong thư mục có tên chứa:
      - "Luật_"
      - "văn bản pháp luật"
      - "Văn bản pháp lý"
      - "văn bản quy phạm pháp luật"
    
    Args:
        law_content_dir: Thư mục gốc chứa các file luật
        verbose: In log chi tiết
    
    Returns:
        Tuple[all_files, law_files_by_category] where:
          - all_files: List đường dẫn tất cả files
          - law_files_by_category: Dict phân loại theo category
    """
    if verbose:
        print(f"🔍 Searching for law files in: {law_content_dir}")
    
    # Tìm tất cả file .docx
    docx_files = glob.glob(os.path.join(law_content_dir, "**", "*.docx"), recursive=True)
    all_files = docx_files
    
    # Lọc theo thư mục có tên phù hợp
    filtered_files = []
    law_folder_keywords = ["Luật_", "văn bản pháp luật", "Văn bản pháp lý", 
                           "văn bản quy phạm pháp luật"]
    
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, law_content_dir)
        path_parts = rel_path.split(os.sep)
        
        is_law_folder = any(
            keyword in part or keyword.lower() in part.lower()
            for part in path_parts
            for keyword in law_folder_keywords
        )
        
        if is_law_folder:
            filtered_files.append(file_path)
            if verbose:
                print(f"   ✅ Found: {rel_path}")
        else:
            if verbose:
                print(f"   ⏭️  Skipped: {rel_path}")
    
    all_files = filtered_files
    
    if verbose:
        print(f"📊 Found {len(all_files)} law files:")
        print(f"   - Total .docx files: {len(docx_files)}")
        print(f"   - Files in law folders: {len(all_files)}")
    
    # Phân loại theo category
    law_files_by_category = {}
    
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, law_content_dir)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) >= 2:
            main_category = path_parts[0]  # VD: "Bất động sản"
            sub_category = path_parts[1]   # VD: "Luật Đất Đai"
            
            if main_category not in law_files_by_category:
                law_files_by_category[main_category] = {}
            if sub_category not in law_files_by_category[main_category]:
                law_files_by_category[main_category][sub_category] = []
            
            file_name = os.path.basename(file_path)
            file_extension = '.' + file_name.split('.')[-1] if '.' in file_name else ''
            
            law_files_by_category[main_category][sub_category].append({
                'file_path': file_path,
                'relative_path': rel_path,
                'file_name': file_name,
                'file_extension': file_extension
            })
    
    # In kết quả phân loại
    if verbose:
        print(f"\n📁 Law files by category:")
        for main_cat, sub_cats in law_files_by_category.items():
            print(f"\n🏛️  {main_cat}:")
            for sub_cat, files in sub_cats.items():
                print(f"   📂 {sub_cat}: {len(files)} files")
                for file_info in files:
                    print(f"      - {file_info['relative_path']}")
    
    return all_files, law_files_by_category


def create_file_paths_list(
    law_files_by_category: Dict[str, Dict[str, List[Dict]]],
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Tạo danh sách file paths từ dictionary phân loại.
    
    Args:
        law_files_by_category: Dictionary files đã phân loại
        verbose: In log chi tiết
    
    Returns:
        List các file paths với metadata
    """
    if verbose:
        print(f"\n📝 Creating law file paths list...")
    
    law_file_paths = []
    
    for main_cat, sub_cats in law_files_by_category.items():
        for sub_cat, files in sub_cats.items():
            for file_info in files:
                law_file_paths.append({
                    'path': file_info['file_path'],
                    'relative_path': file_info['relative_path'],
                    'category': f"{main_cat} - {sub_cat}",
                    'file_name': file_info['file_name'],
                    'extension': file_info['file_extension']
                })
    
    if verbose:
        print(f"✅ Created {len(law_file_paths)} law file paths")
    
    return law_file_paths


def create_category_file_paths(
    law_files_by_category: Dict[str, Dict[str, List[Dict]]]
) -> Dict[str, List[Dict]]:
    """
    Tạo dictionary file paths theo main category.
    
    Args:
        law_files_by_category: Dictionary files đã phân loại
    
    Returns:
        Dictionary file paths theo category chính
    """
    category_file_paths = {}
    
    for main_cat, sub_cats in law_files_by_category.items():
        category_file_paths[main_cat] = []
        
        for sub_cat, files in sub_cats.items():
            for file_info in files:
                category_file_paths[main_cat].append({
                    'path': file_info['file_path'],
                    'relative_path': file_info['relative_path'],
                    'category': f"{main_cat} - {sub_cat}",
                    'file_name': file_info['file_name'],
                    'extension': file_info['file_extension']
                })
    
    return category_file_paths


def find_question_files(
    law_content_dir: str = "law_content",
    verbose: bool = True
) -> Tuple[List[str], Dict[str, Dict[str, List[Dict]]]]:
    """
    Tìm tất cả file câu hỏi Excel (.xlsx) trong thư mục law_content.
    
    Args:
        law_content_dir: Thư mục gốc
        verbose: In log chi tiết
    
    Returns:
        Tuple[xlsx_files, question_files_by_category]
    """
    if verbose:
        print(f"🔍 Searching for question Excel files in: {law_content_dir}")
    
    # Tìm tất cả file .xlsx
    xlsx_files = glob.glob(os.path.join(law_content_dir, "**", "*.xlsx"), recursive=True)
    
    # Lọc bỏ temporary files (bắt đầu bằng ~$)
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith('~$')]
    
    if verbose:
        print(f"📊 Found {len(xlsx_files)} Excel files")
    
    # Phân loại theo thư mục
    question_files_by_category = {}
    
    for file_path in xlsx_files:
        rel_path = os.path.relpath(file_path, law_content_dir)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) >= 2:
            main_category = path_parts[0]
            sub_category = path_parts[1]
            
            if main_category not in question_files_by_category:
                question_files_by_category[main_category] = {}
            if sub_category not in question_files_by_category[main_category]:
                question_files_by_category[main_category][sub_category] = []
            
            question_files_by_category[main_category][sub_category].append({
                'file_path': file_path,
                'relative_path': rel_path,
                'file_name': os.path.basename(file_path)
            })
    
    # In kết quả
    if verbose:
        print(f"\n📁 Question files by category:")
        for main_cat, sub_cats in question_files_by_category.items():
            print(f"\n🏛️  {main_cat}:")
            for sub_cat, files in sub_cats.items():
                print(f"   📂 {sub_cat}: {len(files)} files")
                for file_info in files:
                    print(f"      - {file_info['relative_path']}")
    
    return xlsx_files, question_files_by_category


def read_excel_questions(file_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Đọc file Excel chứa câu hỏi và trả về danh sách câu hỏi.
    
    Args:
        file_path: Đường dẫn file Excel
        verbose: In log chi tiết
    
    Returns:
        List các câu hỏi với structure: {query, positive, negative, id, ...}
    """
    try:
        import pandas as pd
        
        if verbose:
            print(f"   📖 Reading Excel: {os.path.basename(file_path)}")
        
        df = pd.read_excel(file_path)
        
        if verbose:
            print(f"   📊 Found {len(df)} rows, columns: {list(df.columns)}")
        
        questions = []
        
        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip()
        
        # Tìm các cột query, positive, negative
        query_col = None
        positive_col = None
        negative_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'query' in col_lower or 'câu hỏi' in col_lower or 'question' in col_lower:
                query_col = col
            elif 'positive' in col_lower or 'tích cực' in col_lower or 'đúng' in col_lower:
                positive_col = col
            elif 'negative' in col_lower or 'tiêu cực' in col_lower or 'sai' in col_lower:
                negative_col = col
        
        if verbose:
            print(f"   🔍 Detected - Query: {query_col}, "
                  f"Positive: {positive_col}, Negative: {negative_col}")
        
        # Đọc từng dòng
        for idx, row in df.iterrows():
            question_data = {
                'id': f"{os.path.basename(file_path).replace('.xlsx', '')}_Q{idx+1}",
                'query': str(row[query_col]).strip() if query_col and pd.notna(row[query_col]) else "",
                'positive': str(row[positive_col]).strip() if positive_col and pd.notna(row[positive_col]) else "",
                'negative': str(row[negative_col]).strip() if negative_col and pd.notna(row[negative_col]) else "",
                'source_file': os.path.basename(file_path),
                'row_index': idx + 1
            }
            
            # Chỉ thêm nếu có query
            if question_data['query'] and question_data['query'] != 'nan':
                questions.append(question_data)
            else:
                if verbose:
                    print(f"   ⚠️  Skipping row {idx+1}: empty query")
        
        if verbose:
            print(f"   ✅ Extracted {len(questions)} valid questions")
        
        return questions
        
    except ImportError:
        print("   ❌ pandas not installed. Install: pip install pandas openpyxl")
        return []
    except Exception as e:
        print(f"   ❌ Error reading Excel: {e}")
        return []


# Mapping category sang tên folder ngắn
CATEGORY_FOLDER_MAPPING = {
    "Bất động sản": "BDS",
    "Doanh nghiệp_": "DN",
    "Luật Thương Mại": "TM",
    "Quyền dân sự_": "QDS"
}


def get_category_folder_name(main_category: str) -> str:
    """Tạo tên folder từ category chính"""
    return CATEGORY_FOLDER_MAPPING.get(
        main_category,
        main_category.replace(" ", "_").upper()[:3]
    )

