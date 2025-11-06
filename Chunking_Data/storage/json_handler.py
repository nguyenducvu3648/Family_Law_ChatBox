#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Handler Module
===================

Module xử lý lưu/đọc chunks từ JSON files.

Functions:
  - save_chunks_to_json(): Lưu chunks ra JSON
  - load_chunks_from_json(): Đọc chunks từ JSON
  - merge_chunk_files(): Merge nhiều JSON files
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path


def save_chunks_to_json(
    chunks: List[Dict[str, Any]],
    output_path: str,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Lưu chunks ra JSON file.
    
    Args:
        chunks: List chunks cần lưu
        output_path: Đường dẫn file output
        indent: Indent cho pretty print (default: 2)
        ensure_ascii: Escape unicode characters (default: False)
    
    Raises:
        IOError: Nếu không thể ghi file
    """
    # Tạo thư mục nếu chưa tồn tại
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=ensure_ascii, indent=indent)
        
        # Print file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"💾 Saved {len(chunks)} chunks to: {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        
    except Exception as e:
        raise IOError(f"Failed to save chunks: {e}")


def load_chunks_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks từ JSON file.
    
    Args:
        json_path: Đường dẫn file JSON
    
    Returns:
        List chunks
    
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        json.JSONDecodeError: Nếu file không phải valid JSON
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Chunk file not found: {json_path}")
    
    print(f"📖 Loading chunks from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            raise ValueError(f"Expected list of chunks, got {type(chunks)}")
        
        # Validate chunks có content
        valid_chunks = []
        for chunk in chunks:
            if chunk.get('content', '').strip():
                valid_chunks.append(chunk)
            else:
                print(f"   ⚠️  Skipped chunk with empty content: {chunk.get('id', 'unknown')}")
        
        print(f"   ✅ Loaded {len(valid_chunks)} valid chunks")
        return valid_chunks
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file: {e}", e.doc, e.pos)
    except Exception as e:
        raise IOError(f"Failed to load chunks: {e}")


def merge_chunk_files(
    file_paths: List[str],
    remove_duplicates: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge nhiều chunk JSON files thành 1 list.
    
    Args:
        file_paths: List đường dẫn các file cần merge
        remove_duplicates: Loại bỏ duplicates theo chunk ID
        verbose: In log chi tiết
    
    Returns:
        List chunks đã merge
    """
    if verbose:
        print(f"📦 Merging {len(file_paths)} files...")
    
    all_chunks = []
    seen_ids = set()
    
    for i, file_path in enumerate(file_paths, 1):
        basename = os.path.basename(file_path)
        
        if verbose:
            print(f"\n[{i}/{len(file_paths)}] Loading: {basename}")
        
        try:
            chunks = load_chunks_from_json(file_path)
            
            if remove_duplicates:
                # Remove duplicates by chunk ID
                new_chunks = []
                duplicates = 0
                for chunk in chunks:
                    chunk_id = chunk.get('id')
                    if chunk_id not in seen_ids:
                        new_chunks.append(chunk)
                        seen_ids.add(chunk_id)
                    else:
                        duplicates += 1
                
                all_chunks.extend(new_chunks)
                
                if verbose:
                    print(f"   ✅ Added {len(new_chunks)} chunks "
                          f"({duplicates} duplicates skipped)")
            else:
                all_chunks.extend(chunks)
                if verbose:
                    print(f"   ✅ Added {len(chunks)} chunks")
                    
        except Exception as e:
            if verbose:
                safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                print(f"   ❌ Error: {safe_error}")
            continue
    
    if verbose:
        print(f"\n✅ Total chunks after merge: {len(all_chunks)}")
    
    return all_chunks


def get_file_stats(json_path: str) -> Dict[str, Any]:
    """
    Lấy thống kê về JSON file.
    
    Args:
        json_path: Đường dẫn file JSON
    
    Returns:
        Dict stats với keys:
          - file_size_mb
          - total_chunks
          - avg_chunk_length
          - law_ids (unique)
          - categories (unique)
    """
    chunks = load_chunks_from_json(json_path)
    
    file_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
    
    # Calculate avg chunk length
    total_length = sum(len(c.get('content', '')) for c in chunks)
    avg_length = total_length / len(chunks) if chunks else 0
    
    # Extract unique law_ids và categories
    law_ids = set()
    categories = set()
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        law_id = metadata.get('law_id')
        if law_id:
            law_ids.add(law_id)
        
        category = metadata.get('category')
        if category:
            categories.add(category)
    
    return {
        'file_size_mb': round(file_size, 2),
        'total_chunks': len(chunks),
        'avg_chunk_length': round(avg_length, 1),
        'unique_law_ids': sorted(list(law_ids)),
        'unique_categories': sorted(list(categories))
    }

