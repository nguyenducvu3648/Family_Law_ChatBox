#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Chunks Script
===================

Script merge nhiều chunk JSON files thành 1 file lớn.

Usage:
    # Merge tất cả files trong directory
    python -m Chunking_Data.scripts.merge_chunks --directory data/BDS
    
    # Merge files cụ thể
    python -m Chunking_Data.scripts.merge_chunks --files file1.json file2.json
"""

import argparse
import os
import glob
from datetime import datetime
from pathlib import Path

from ..storage.json_handler import merge_chunk_files, save_chunks_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Merge nhiều chunk JSON files thành 1 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Merge tất cả files trong directory
  python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS
  
  # Merge files cụ thể
  python -m Chunking_Data.scripts.merge_chunks --files file1.json file2.json --output merged.json
  
  # Merge với pattern
  python -m Chunking_Data.scripts.merge_chunks --directory data/ --pattern "BDS_*_chunk_*.json"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--directory",
        help="Thư mục chứa chunk files"
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        help="Danh sách files cụ thể"
    )
    
    parser.add_argument(
        "--pattern",
        default="*_chunk_*.json",
        help="Pattern để tìm files (với --directory). Default: *_chunk_*.json"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path. Nếu không chỉ định, sẽ auto-generate"
    )
    
    parser.add_argument(
        "--category",
        help="Category name (dùng cho auto-generate output filename)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)"
    )
    
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Giữ lại chunks duplicate (mặc định: remove duplicates by ID)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: không ghi file output"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In log chi tiết"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📦 MERGE CHUNK FILES")
    print("=" * 80)
    
    # Determine input files
    if args.directory:
        print(f"📁 Searching in: {args.directory}")
        print(f"   Pattern: {args.pattern}")
        chunk_files = glob.glob(os.path.join(args.directory, args.pattern))
        chunk_files.sort()
    else:
        chunk_files = args.files
    
    if not chunk_files:
        print("❌ No chunk files found!")
        return
    
    print(f"✅ Found {len(chunk_files)} files:")
    for i, f in enumerate(chunk_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        basename = os.path.basename(f)
        print(f"   {i}. {basename} ({size_mb:.2f} MB)")
    
    print("=" * 80)
    
    # Merge
    remove_dups = not args.keep_duplicates
    all_chunks = merge_chunk_files(
        file_paths=chunk_files,
        remove_duplicates=remove_dups,
        verbose=args.verbose
    )
    
    if not all_chunks:
        print("❌ No chunks to save!")
        return
    
    # Generate output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate
        timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
        if args.category:
            output_filename = f"{args.category}_merged_chunk_{timestamp}.json"
        else:
            output_filename = f"merged_chunk_{timestamp}.json"
        output_path = os.path.join(args.output_dir, output_filename)
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN: Would save {len(all_chunks)} chunks to: {output_path}")
        return
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_chunks_to_json(all_chunks, output_path)
    
    # Stats
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    # Breakdown by law_id
    law_ids = {}
    for chunk in all_chunks:
        law_id = chunk.get('metadata', {}).get('law_id', 'Unknown')
        law_ids[law_id] = law_ids.get(law_id, 0) + 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ MERGE COMPLETED!")
    print(f"{'='*80}")
    print(f"📦 Total chunks: {len(all_chunks)}")
    print(f"📁 File size: {file_size_mb:.2f} MB")
    print(f"📄 Output: {output_path}")
    
    if len(law_ids) > 1:
        print(f"\n📊 Breakdown by law_id:")
        for law_id, count in sorted(law_ids.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {law_id}: {count} chunks")
    
    print(f"\n💡 Next step:")
    category = args.category or 'YOUR_CATEGORY'
    print(f"   python -m Chunking_Data.scripts.upload_qdrant --chunk-file \"{output_path}\" --category {category}")


if __name__ == "__main__":
    main()

