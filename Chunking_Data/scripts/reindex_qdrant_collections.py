#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reindex Qdrant Collections - Tạo/Cập nhật payload indexes cho collections hiện có

Dựa trên cấu trúc metadata thực tế từ data mẫu:
- law_no, law_title, law_id
- issued_date, effective_date, expiry_date
- signer
- chapter, chapter_number, chapter_title
- section
- article_no, article_title
- clause_no, clause_intro
- point_id, point_letter
- exact_citation

Usage:
  - Set env QDRANT_URL and QDRANT_API_KEY (optional)
  - Run:
      python reindex_qdrant_collections.py --collection BAAI_bge_m3_LHN2014
    or multiple:
      python reindex_qdrant_collections.py --collection c1 --collection c2
    or all collections:
      python reindex_qdrant_collections.py --all
"""

import os
import sys
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PayloadSchemaType
except Exception as e:
    print(f"[ERROR] Missing qdrant_client: {e}")
    sys.exit(1)


# Định nghĩa các field cần index dựa trên cấu trúc metadata thực tế
INDEX_FIELDS = {
    # Thông tin luật
    "metadata.law_id": PayloadSchemaType.KEYWORD,
    "metadata.law_title": PayloadSchemaType.KEYWORD,
    "metadata.law_no": PayloadSchemaType.KEYWORD,

    # Thời gian
    "metadata.issued_date": PayloadSchemaType.DATETIME,
    "metadata.effective_date": PayloadSchemaType.DATETIME,
    "metadata.expiry_date": PayloadSchemaType.DATETIME,

    # Người ký
    "metadata.signer": PayloadSchemaType.KEYWORD,

    # Cấu trúc pháp điển
    "metadata.chapter": PayloadSchemaType.KEYWORD,
    "metadata.chapter_number": PayloadSchemaType.INTEGER,
    "metadata.chapter_title": PayloadSchemaType.KEYWORD,
    "metadata.section": PayloadSchemaType.KEYWORD,

    # Điều
    "metadata.article_no": PayloadSchemaType.INTEGER,
    "metadata.article_title": PayloadSchemaType.KEYWORD,

    # Khoản
    "metadata.clause_no": PayloadSchemaType.INTEGER,
    "metadata.clause_intro": PayloadSchemaType.TEXT,  # Có thể chứa text dài

    # Điểm
    "metadata.point_id": PayloadSchemaType.KEYWORD,
    "metadata.point_letter": PayloadSchemaType.KEYWORD,

    # Trích dẫn chính xác
    "metadata.exact_citation": PayloadSchemaType.KEYWORD,

    # Nguồn (nếu có)
    "metadata.source_category": PayloadSchemaType.KEYWORD,
    "metadata.source_file_name": PayloadSchemaType.KEYWORD,
    "metadata.source_file": PayloadSchemaType.KEYWORD,
    "metadata.chunk_index": PayloadSchemaType.INTEGER,

    # Các field bổ sung có thể có
    "metadata.category": PayloadSchemaType.KEYWORD,
    "metadata.subcategory": PayloadSchemaType.KEYWORD,
}


def get_qdrant_client() -> QdrantClient:
    """Kết nối với Qdrant server."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY") or None

    if not url:
        # Fallback to localhost if not provided
        url = "http://localhost:6333"
        print(f"[WARNING] QDRANT_URL not set, using default: {url}")

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=300.0,
            grpc_port=6334,
        )
        print("[OK] Qdrant connected successfully")
        return client
    except Exception as e:
        print(f"[ERROR] Cannot connect to Qdrant: {e}")
        sys.exit(1)


def get_all_collections(client: QdrantClient) -> List[str]:
    """Lấy danh sách tất cả collections."""
    try:
        collections = client.get_collections().collections
        return [c.name for c in collections]
    except Exception as e:
        print(f"[ERROR] Cannot get collections: {e}")
        return []


def reindex_collection(client: QdrantClient, collection_name: str) -> Dict[str, int]:
    """
    Tạo/cập nhật payload indexes cho một collection.

    Returns:
        Dict với số lượng created, skipped, failed
    """
    print(f"\n[REINDEX] Reindexing collection: {collection_name}")

    # Validate collection exists
    all_collections = get_all_collections(client)
    if collection_name not in all_collections:
        print(f"[ERROR] Collection '{collection_name}' not found. Available: {sorted(all_collections)}")
        return {"created": 0, "skipped": 0, "failed": 0}

    # Get collection info để hiển thị
    try:
        info = client.get_collection(collection_name)
        points_count = info.points_count or 0
        print(f"   [INFO] Collection has {points_count} points")
    except Exception as e:
        print(f"   [WARNING] Cannot get collection info: {e}")

    created = 0
    skipped = 0
    failed = 0

    for field_name, schema_type in INDEX_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            print(f"   [+] Indexed: {field_name} ({schema_type})")
            created += 1

        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "already indexed" in error_msg:
                print(f"   [SKIP] Already indexed: {field_name}")
                skipped += 1
            else:
                print(f"   [FAIL] Failed to index '{field_name}': {e}")
                failed += 1

    print(f"[DONE] Created={created}, Skipped={skipped}, Failed={failed}")
    return {"created": created, "skipped": skipped, "failed": failed}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reindex Qdrant collections with updated payload indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reindex specific collection
  python reindex_qdrant_collections.py --collection BAAI_bge_m3_LHN2014

  # Reindex multiple collections
  python reindex_qdrant_collections.py --collection c1 --collection c2

  # Reindex all collections
  python reindex_qdrant_collections.py --all

  # Dry run (show what would be done)
  python reindex_qdrant_collections.py --all --dry-run
        """
    )

    parser.add_argument(
        "--collection",
        action="append",
        help="Collection name (can be specified multiple times)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Reindex all available collections"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating indexes"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Get collections to process
    collections = []
    if args.all:
        client = get_qdrant_client()
        collections = get_all_collections(client)
        if not collections:
            print("[ERROR] No collections found")
            return
        print(f"[INFO] Found {len(collections)} collections: {collections}")
    elif args.collection:
        collections = args.collection
    else:
        print("[ERROR] Please specify --collection or --all")
        print("[TIP] Run with --help for usage examples")
        return

    if args.dry_run:
        print("[DRY RUN] No actual changes will be made")
        print(f"[INFO] Would process collections: {collections}")
        print(f"[TARGET] Index fields: {list(INDEX_FIELDS.keys())}")
        return

    # Connect to Qdrant
    client = get_qdrant_client()

    # Process collections
    total_stats = {"created": 0, "skipped": 0, "failed": 0}

    for collection in collections:
        stats = reindex_collection(client, collection)
        total_stats["created"] += stats["created"]
        total_stats["skipped"] += stats["skipped"]
        total_stats["failed"] += stats["failed"]

    # Summary
    print("""
[SUCCESS] Reindexing completed!""")
    print(f"[STATS] Total: Created={total_stats['created']}, Skipped={total_stats['skipped']}, Failed={total_stats['failed']}")

    if total_stats["failed"] > 0:
        print(f"[WARNING] {total_stats['failed']} indexes failed to create. Check logs above.")
    else:
        print("[SUCCESS] All collections reindexed successfully!")


if __name__ == "__main__":
    main()
