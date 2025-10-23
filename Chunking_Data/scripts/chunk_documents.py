#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk Documents Script
=======================

Script chunk văn bản luật thành chunks.

Usage:
    # Chunk theo category
    python -m Chunking_Data.scripts.chunk_documents --category BDS
    
    # Chunk file cụ thể
    python -m Chunking_Data.scripts.chunk_documents --file path/to/law.docx
    
    # Chunk tất cả
    python -m Chunking_Data.scripts.chunk_documents --all
"""

import argparse
import os
import json
import sys
from datetime import datetime
from pathlib import Path

from ..pipeline.chunking_pipeline import ChunkingPipeline, validate_chunks
from ..storage.json_handler import save_chunks_to_json
from ..evaluation.ai_reviewer import (
    build_review_payload,
    call_gemini_review,
    save_issues_report,
    print_review_summary
)

# For .env loading
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_file_paths(category: str = None, data_files_dir: str = "data_files"):
    """Load file paths từ JSON"""
    if category:
        # Load category-specific
        from ..pipeline.file_discovery import get_category_folder_name
        folder_name = get_category_folder_name(category)
        json_path = os.path.join(data_files_dir, folder_name, f"{folder_name.lower()}_file_paths.json")
    else:
        # Load all
        json_path = os.path.join(data_files_dir, "law_file_paths.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"File paths not found: {json_path}\n"
            f"Run find_files.py first!"
        )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk văn bản luật thành chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Chunk theo category
  python -m Chunking_Data.scripts.chunk_documents --category BDS
  
  # Chunk file cụ thể
  python -m Chunking_Data.scripts.chunk_documents --file "path/to/law.docx"
  
  # Chunk tất cả với validation
  python -m Chunking_Data.scripts.chunk_documents --all --validate
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--category",
        help="Chunk theo category (BDS, DN, TM, QDS)"
    )
    input_group.add_argument(
        "--file",
        help="Chunk file cụ thể"
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Chunk tất cả categories"
    )
    
    # Metadata options
    parser.add_argument(
        "--law-no",
        default="",
        help="Số hiệu luật (default: empty)"
    )
    parser.add_argument(
        "--issued-date",
        default="",
        help="Ngày ban hành (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--effective-date",
        default="",
        help="Ngày có hiệu lực (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--signer",
        default="",
        help="Người ký"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--output-name",
        help="Custom output filename (without extension)"
    )
    
    # Processing options
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate chunks sau khi tạo"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: không ghi file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In log chi tiết"
    )
    
    # AI Review options
    parser.add_argument(
        "--AI",
        action="store_true",
        help="Bật AI review với Gemini (default: OFF)"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (hoặc set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--sample-excerpts",
        type=int,
        default=2000,
        help="Tổng ký tự excerpts cho AI review (default: 2000)"
    )
    parser.add_argument(
        "--max-chunks-sample",
        type=int,
        default=50,
        help="Số chunks tối đa gửi AI review (default: 50)"
    )
    parser.add_argument(
        "--max-files-sample",
        type=int,
        default=2,
        help="Số files tối đa lấy raw text cho AI review (default: 2)"
    )
    parser.add_argument(
        "--strict-ok-only",
        action="store_true",
        help="Chỉ lưu chunks nếu AI confirm OK (chỉ khi --AI)"
    )
    
    args = parser.parse_args()
    
    # Load .env if available
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    print("=" * 80)
    print("✂️  CHUNK DOCUMENTS")
    if args.AI:
        print("🤖 WITH AI REVIEW ENABLED")
    print("=" * 80)
    
    # Determine input files
    if args.file:
        # Single file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        
        file_name = os.path.basename(args.file)
        file_paths = [{
            'path': args.file,
            'file_name': file_name,
            'category': 'Single File'
        }]
        output_prefix = file_name.replace('.docx', '').replace('.doc', '').replace(' ', '_')[:30]
        
    elif args.category:
        # Category
        file_paths = load_file_paths(category=args.category)
        output_prefix = args.category
        
    else:  # args.all
        # All categories
        file_paths = load_file_paths(category=None)
        output_prefix = "all"
    
    print(f"📁 Files to process: {len(file_paths)}")
    print(f"📂 Output directory: {args.output_dir}")
    
    if args.dry_run:
        print(f"🔍 DRY RUN MODE - No files will be written")
    
    if args.AI:
        print(f"🤖 AI Review:")
        print(f"   - Max files sample: {args.max_files_sample}")
        print(f"   - Max chunks sample: {args.max_chunks_sample}")
        print(f"   - Sample excerpts: {args.sample_excerpts} chars")
        if args.strict_ok_only:
            print(f"   - Strict OK only: ENABLED")
    
    print("=" * 80)
    
    # Create pipeline
    pipeline = ChunkingPipeline(verbose=args.verbose)
    
    # Process files
    chunks, summary = pipeline.process_files(
        file_paths=file_paths,
        default_law_no=args.law_no,
        default_issued_date=args.issued_date,
        default_effective_date=args.effective_date,
        default_signer=args.signer
    )
    
    if not chunks:
        print("❌ No chunks created!")
        return
    
    # Validate if requested
    if args.validate:
        validation = validate_chunks(chunks, verbose=args.verbose)
        if validation["invalid_chunks"] > 0:
            print(f"\n⚠️  Found {validation['invalid_chunks']} invalid chunks!")
        else:
            print(f"\n✅ All {validation['valid_chunks']} chunks are valid!")
    
    # AI Review if requested
    ai_review_passed = True
    if args.AI:
        print(f"\n{'='*80}")
        print(f"🤖 AI REVIEW")
        print(f"{'='*80}")
        
        # Collect raw texts for AI review
        raw_texts = []
        if args.max_files_sample > 0:
            print(f"📖 Collecting raw texts for AI review (max {args.max_files_sample} files)...")
            files_sampled = 0
            
            for file_info in file_paths:
                if files_sampled >= args.max_files_sample:
                    break
                
                file_path = file_info['path']
                if os.path.exists(file_path):
                    try:
                        from ..core.docx_reader import read_docx
                        raw_text = read_docx(file_path)
                        if raw_text and len(raw_text.strip()) > 100:
                            # Limit to 5k chars per file
                            raw_texts.append(raw_text[:5000])
                            files_sampled += 1
                            if args.verbose:
                                print(f"   ✅ Sampled file {files_sampled}: {file_info['file_name']} ({len(raw_text)} chars)")
                    except Exception as e:
                        if args.verbose:
                            print(f"   ⚠️  Failed to read {file_info['file_name']}: {e}")
                        continue
            
            print(f"   ✅ Collected excerpts from {len(raw_texts)} files (total ~{sum(len(t) for t in raw_texts)} chars)")
        
        # Build review payload
        print(f"🔨 Building AI review payload...")
        payload = build_review_payload(
            chunks=chunks,
            summary=summary,
            raw_texts=raw_texts,
            sample_excerpts_chars=args.sample_excerpts,
            max_chunks_sample=args.max_chunks_sample
        )
        
        print(f"📤 Calling Gemini AI for review...")
        print(f"   Sending: {len(payload['chunks_preview'])} sampled chunks + {len(raw_texts)} file excerpts")
        print(f"   Payload size: ~{len(str(payload)) // 1000}KB")
        
        try:
            review = call_gemini_review(payload, args.api_key)
            
            # Print review summary
            print_review_summary(review, verbose=args.verbose)
            
            # Check if passed
            status = review.get("status", "issues_found")
            issues = review.get("issues", [])
            
            if status != "ok" or issues:
                ai_review_passed = False
                
                # Generate issues filename
                timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
                issues_filename = f"{output_prefix}_chunk_{timestamp}.issues.json"
                
                if args.category and not args.file:
                    issues_dir = os.path.join(args.output_dir, args.category)
                else:
                    issues_dir = args.output_dir
                
                os.makedirs(issues_dir, exist_ok=True)
                issues_path = os.path.join(issues_dir, issues_filename)
                
                # Save issues report
                save_issues_report(review, issues_path)
                print(f"💾 Issues report saved: {issues_path}")
                
                if args.strict_ok_only:
                    print(f"\n❌ AI Review FAILED - --strict-ok-only enabled, not saving chunks!")
                    return
            else:
                print(f"✅ AI Review PASSED!")
                
        except Exception as e:
            print(f"❌ AI Review ERROR: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            
            if args.strict_ok_only:
                print(f"\n❌ --strict-ok-only enabled, cannot proceed without AI review!")
                return
            else:
                print(f"⚠️  Proceeding without AI confirmation...")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN: Would save {len(chunks)} chunks")
        return
    
    # Generate output filename
    if args.output_name:
        output_filename = f"{args.output_name}.json"
    else:
        timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
        output_filename = f"{output_prefix}_chunk_{timestamp}.json"
    
    # Create output directory
    if args.category and not args.file:
        # Category subfolder
        final_output_dir = os.path.join(args.output_dir, args.category)
    else:
        final_output_dir = args.output_dir
    
    os.makedirs(final_output_dir, exist_ok=True)
    output_path = os.path.join(final_output_dir, output_filename)
    
    # Save chunks
    save_chunks_to_json(chunks, output_path)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ CHUNKING COMPLETED!")
    print(f"{'='*80}")
    print(f"📦 Total chunks: {len(chunks)}")
    print(f"📊 Statistics:")
    print(f"   - Articles: {summary.get('articles', 0)}")
    print(f"   - Clauses: {summary.get('clauses', 0)}")
    print(f"   - Points: {summary.get('points', 0)}")
    print(f"   - Chapters: {len(summary.get('chapters_seen', []))}")
    print(f"📁 Output: {output_path}")
    print(f"\n💡 Next step:")
    print(f"   python -m Chunking_Data.scripts.upload_qdrant --chunk-file \"{output_path}\" --category YOUR_CATEGORY")


if __name__ == "__main__":
    main()

