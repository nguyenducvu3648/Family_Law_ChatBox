#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point cho Chunking_Data package.

Usage:
    python -m Chunking_Data
    python -m Chunking_Data --help
    python -m Chunking_Data --config
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Chunking Data Package - Vietnamese Legal Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AVAILABLE SCRIPTS:
  find_files        - Tìm và catalog files luật
  chunk_documents   - Chunk văn bản luật thành chunks
  merge_chunks      - Merge nhiều chunk files
  upload_qdrant     - Upload chunks lên Qdrant

USAGE EXAMPLES:
  # Show configuration
  python -m Chunking_Data --config
  
  # Run scripts
  python -m Chunking_Data.scripts.find_files
  python -m Chunking_Data.scripts.chunk_documents --category BDS
  python -m Chunking_Data.scripts.merge_chunks --directory data/BDS
  python -m Chunking_Data.scripts.upload_qdrant --chunk-file data/BDS.json --category BDS

For more details, see README.md
        """
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration summary"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration (check .env)"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show package version"
    )
    
    args = parser.parse_args()
    
    if args.config:
        from .config import print_config_summary
        print_config_summary()
        return 0
    
    if args.validate_config:
        from .config import validate_config
        if validate_config():
            print("✅ Configuration is valid")
            return 0
        else:
            print("❌ Configuration is invalid")
            print("\nPlease check:")
            print("  1. Create .env file (see env.example)")
            print("  2. Set QDRANT_URL and QDRANT_API_KEY")
            return 1
    
    if args.version:
        from . import __version__
        print(f"Chunking_Data version {__version__}")
        return 0
    
    # Default: print help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

