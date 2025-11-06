#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunking Pipeline Module
=========================

High-level pipeline để chunk nhiều văn bản luật theo batch.

Features:
  - Batch processing nhiều files
  - Auto law_id generation
  - Error handling và retry
  - Progress tracking
  - Statistics collection
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..core.docx_reader import read_docx
from ..core.law_chunker import chunk_law_document
from ..core.law_id_generator import generate_law_id


class ChunkingPipeline:
    """
    Pipeline để chunk nhiều văn bản luật.
    
    Usage:
        pipeline = ChunkingPipeline(verbose=True)
        chunks, stats = pipeline.process_files(file_paths)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize pipeline.
        
        Args:
            verbose: In log chi tiết
        """
        self.verbose = verbose
        self.total_chunks = 0
        self.successful_files = 0
        self.failed_files = 0
        self.warnings = []
    
    def process_single_file(
        self,
        file_path: str,
        law_no: str = "",
        law_title: Optional[str] = None,
        law_id: Optional[str] = None,
        issued_date: str = "",
        effective_date: str = "",
        expiry_date: Optional[str] = None,
        signer: str = ""
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Chunk một file đơn lẻ.
        
        Args:
            file_path: Đường dẫn file
            law_no: Số hiệu luật
            law_title: Tên luật (auto từ filename nếu None)
            law_id: ID luật (auto generate nếu None)
            issued_date: Ngày ban hành
            effective_date: Ngày có hiệu lực
            expiry_date: Ngày hết hiệu lực
            signer: Người ký
        
        Returns:
            Tuple[chunks, stats]
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu không đọc được content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_name = os.path.basename(file_path)
        
        if self.verbose:
            safe_name = file_name.encode('ascii', 'ignore').decode('ascii') or file_name
            print(f"\n📄 Processing: {safe_name}")
        
        # 1. Đọc file
        if self.verbose:
            print("   📖 Reading file...")
        raw_text = read_docx(file_path, verbose=self.verbose)
        
        if not raw_text or len(raw_text.strip()) < 100:
            raise ValueError(f"File content too short: {len(raw_text)} chars")
        
        # 2. Tự động tạo law_id nếu chưa có
        if law_id is None:
            law_id = generate_law_id(file_name)
            if self.verbose:
                print(f"   🆔 Generated law_id: {law_id}")
        
        # 3. Tự động tạo law_title nếu chưa có
        if law_title is None:
            law_title = file_name.replace('.docx', '').replace('.doc', '').strip()
        
        # 4. Chunk document
        if self.verbose:
            print("   ✂️  Chunking document...")
        
        chunks, chunk_stats = chunk_law_document(
            text=raw_text,
            law_id=law_id,
            law_no=law_no,
            law_title=law_title,
            issued_date=issued_date,
            effective_date=effective_date,
            expiry_date=expiry_date,
            signer=signer,
            verbose=self.verbose
        )
        
        if not chunks:
            raise ValueError("No chunks created from file")
        
        return chunks, chunk_stats
    
    def process_files(
        self,
        file_paths: List[Dict[str, str]],
        default_law_no: str = "",
        default_law_title: str = "",
        default_law_id: str = "",
        default_issued_date: str = "",
        default_effective_date: str = "",
        default_signer: str = ""
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Chunk nhiều files theo batch.

        Args:
            file_paths: List file paths (từ find_law_files)
                Format: [{'path': ..., 'file_name': ..., 'category': ...}, ...]
            default_law_no: Số hiệu luật mặc định
            default_law_title: Tên luật mặc định (nếu không thì auto từ filename)
            default_law_id: ID luật mặc định (nếu không thì auto generate)
            default_issued_date: Ngày ban hành mặc định
            default_effective_date: Ngày hiệu lực mặc định
            default_signer: Người ký mặc định

        Returns:
            Tuple[all_chunks, summary_stats]
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 Starting batch chunking: {len(file_paths)} files")
            print(f"{'='*80}")
        
        all_chunks = []
        chapters_seen_all = []
        citations_all = []
        total_stats = {
            'articles': 0,
            'article_intro': 0,
            'clauses': 0,
            'points': 0
        }
        
        for i, file_info in enumerate(file_paths, 1):
            file_path = file_info['path']
            file_name = file_info['file_name']
            category = file_info.get('category', 'Unknown')
            
            if self.verbose:
                safe_name = file_name.encode('ascii', 'ignore').decode('ascii') or file_name
                safe_cat = str(category).encode('ascii', 'ignore').decode('ascii') or str(category)
                print(f"\n[{i}/{len(file_paths)}] 📂 {safe_cat}")
                print(f"   📄 {safe_name}")
            
            try:
                # Process file
                chunks, chunk_stats = self.process_single_file(
                    file_path=file_path,
                    law_no=default_law_no,
                    law_title=default_law_title or None,  # Use default if provided, else auto từ filename
                    law_id=default_law_id or None,         # Use default if provided, else auto generate
                    issued_date=default_issued_date,
                    effective_date=default_effective_date,
                    expiry_date=None,
                    signer=default_signer
                )
                
                # Collect chunks
                all_chunks.extend(chunks)
                
                # Collect stats
                for key in total_stats:
                    total_stats[key] += chunk_stats.get(key, 0)
                
                # Collect chapters và citations
                for chapter in chunk_stats.get('chapters_seen', []):
                    if chapter and chapter not in chapters_seen_all:
                        chapters_seen_all.append(chapter)
                
                citations_all.extend(chunk_stats.get('citations', []))
                
                self.successful_files += 1
                
                if self.verbose:
                    print(f"   ✅ Success: {len(chunks)} chunks")
                
            except Exception as e:
                self.failed_files += 1
                safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                error_msg = f"Failed to process {file_name}: {safe_error}"
                self.warnings.append(error_msg)
                
                if self.verbose:
                    print(f"   ❌ Error: {safe_error}")
                
                continue
        
        # Summary
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📊 CHUNKING SUMMARY")
            print(f"{'='*80}")
            print(f"✅ Successful: {self.successful_files} files")
            print(f"❌ Failed: {self.failed_files} files")
            print(f"📦 Total chunks: {len(all_chunks)}")
            
            if all_chunks:
                avg_len = sum(len(c['content']) for c in all_chunks) / len(all_chunks)
                print(f"📏 Average chunk length: {avg_len:.0f} chars")
            
            print(f"\n📈 Statistics:")
            print(f"   - Articles: {total_stats['articles']}")
            print(f"   - Article Intro: {total_stats['article_intro']}")
            print(f"   - Clauses: {total_stats['clauses']}")
            print(f"   - Points: {total_stats['points']}")
            print(f"   - Chapters: {len(chapters_seen_all)}")
            print(f"   - Citations: {len(citations_all)}")
        
        # Create summary
        summary = {
            'chapters_seen': chapters_seen_all,
            'articles': total_stats['articles'],
            'article_intro': total_stats['article_intro'],
            'clauses': total_stats['clauses'],
            'points': total_stats['points'],
            'citations': citations_all,
            'warnings': self.warnings,
            'total_chunks': len(all_chunks),
            'successful_files': self.successful_files,
            'failed_files': self.failed_files
        }
        
        return all_chunks, summary
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.total_chunks = 0
        self.successful_files = 0
        self.failed_files = 0
        self.warnings = []


def validate_chunks(
    chunks: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate chunks sau khi tạo.
    
    Args:
        chunks: List chunks cần validate
        verbose: In chi tiết từng lỗi
    
    Returns:
        Dict validation results với keys:
          - total_chunks
          - valid_chunks
          - invalid_chunks
          - issues
    """
    if verbose:
        print(f"\n🔍 Validating {len(chunks)} chunks...")
    
    validation_results = {
        "total_chunks": len(chunks),
        "valid_chunks": 0,
        "invalid_chunks": 0,
        "issues": []
    }
    
    required_fields = ["id", "content", "metadata"]
    required_metadata = ["law_id", "article_no", "exact_citation"]
    
    for i, chunk in enumerate(chunks):
        is_valid = True
        issues = []
        
        # Check required fields
        for field in required_fields:
            if field not in chunk:
                issues.append(f"Missing field: {field}")
                is_valid = False
        
        # Check metadata
        if "metadata" in chunk:
            metadata = chunk["metadata"]
            for field in required_metadata:
                if field not in metadata:
                    issues.append(f"Missing metadata: {field}")
                    is_valid = False
            
            # Check content length
            if len(chunk.get("content", "")) < 50:
                issues.append("Content too short (< 50 chars)")
                is_valid = False
            
            # Check ID format
            chunk_id = chunk.get("id", "")
            if not chunk_id or len(chunk_id.split("-")) < 2:
                issues.append("Invalid ID format")
                is_valid = False
        
        if is_valid:
            validation_results["valid_chunks"] += 1
        else:
            validation_results["invalid_chunks"] += 1
            validation_results["issues"].append({
                "chunk_index": i,
                "chunk_id": chunk.get("id", "unknown"),
                "issues": issues
            })
            
            if verbose:
                print(f"  ⚠️  Chunk {i} ({chunk.get('id', 'unknown')}): {', '.join(issues)}")
    
    if verbose:
        print(f"✅ Valid: {validation_results['valid_chunks']}")
        print(f"❌ Invalid: {validation_results['invalid_chunks']}")
    
    return validation_results

