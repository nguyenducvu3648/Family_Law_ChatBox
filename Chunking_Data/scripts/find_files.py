#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find Law Files Script
=====================

Script tìm và catalog các file luật trong law_content.

Usage:
    python -m Chunking_Data.scripts.find_files [--output OUTPUT_DIR]
"""

import argparse
import os
import json
from pathlib import Path

from ..pipeline.file_discovery import (
    find_law_files,
    create_file_paths_list,
    create_category_file_paths,
    get_category_folder_name
)


def main():
    parser = argparse.ArgumentParser(
        description="Tìm và catalog các file luật",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--law-content-dir",
        default="law_content",
        help="Thư mục gốc chứa files luật (default: law_content)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data_files",
        help="Thư mục output cho JSON files (default: data_files)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In log chi tiết"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 FIND LAW FILES")
    print("=" * 80)
    print(f"Source: {args.law_content_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # 1. Find files
    all_files, law_files_by_category = find_law_files(
        law_content_dir=args.law_content_dir,
        verbose=args.verbose
    )
    
    if not all_files:
        print("❌ No law files found!")
        return
    
    # 2. Create file paths list
    law_file_paths = create_file_paths_list(
        law_files_by_category,
        verbose=args.verbose
    )
    
    # 3. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Save consolidated file paths
    output_file = os.path.join(args.output_dir, "law_file_paths.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(law_file_paths, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved {len(law_file_paths)} file paths to: {output_file}")
    
    # 5. Save category-specific file paths
    category_file_paths = create_category_file_paths(law_files_by_category)
    
    print(f"\n📁 Saving file paths by category...")
    for main_cat, file_paths in category_file_paths.items():
        if not file_paths:
            continue
        
        folder_name = get_category_folder_name(main_cat)
        folder_path = os.path.join(args.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        file_name = f"{folder_name.lower()}_file_paths.json"
        output_path = os.path.join(folder_path, file_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_paths, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 {main_cat}: {len(file_paths)} files → {output_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED!")
    print(f"{'='*80}")
    print(f"📊 Total files found: {len(all_files)}")
    print(f"📂 Categories: {len(category_file_paths)}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

