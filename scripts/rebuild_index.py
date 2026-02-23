#!/usr/bin/env python3
"""Rebuild ChromaDB index from AsciiDoc files.

This script processes all .adoc files and rebuilds the vector index.

Usage:
    python scripts/rebuild_index.py           # Rebuild index
    python scripts/rebuild_index.py --dry-run # Preview chunks without storing
"""
import sys
import argparse
import logging
from pathlib import Path

# Configure logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.document_processor import DocumentProcessor
from src.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild ChromaDB index from AsciiDoc files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunks without storing to ChromaDB"
    )
    parser.add_argument(
        "--adoc-path",
        type=str,
        default=None,
        help="Override ADOC_FILES_PATH from config"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )
    args = parser.parse_args()

    # Determine paths
    adoc_path = args.adoc_path or Config.ADOC_FILES_PATH
    chromadb_path = Config.CHROMADB_PATH

    print(f"📁 ADOC files path: {adoc_path}")
    print(f"💾 ChromaDB path: {chromadb_path}")
    print(f"🤖 Embedding model: {Config.EMBEDDING_MODEL}")
    print(f"🌐 Ollama host: {Config.OLLAMA_HOST}")
    print()

    # Check if adoc path exists
    if not Path(adoc_path).exists():
        print(f"❌ Error: ADOC files path does not exist: {adoc_path}")
        print("   Please create the directory and add .adoc files, or set ADOC_FILES_PATH in .env")
        sys.exit(1)

    # Test Ollama connection
    print("🔌 Testing Ollama connection...")
    try:
        import ollama
        ollama.list()
        print("✅ Ollama connection successful")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama at {Config.OLLAMA_HOST}: {e}")
        sys.exit(1)
    print()

    # Initialize processor
    processor = DocumentProcessor(
        chromadb_path=chromadb_path,
        embedding_model=Config.EMBEDDING_MODEL,
        ollama_host=Config.OLLAMA_HOST
    )

    if args.dry_run:
        print("🔍 DRY RUN - Preview mode (no changes will be made)")
        print("=" * 60)
        
        # Get stats without storing
        adoc_files = processor.load_adoc_files(adoc_path)
        print(f"Found {len(adoc_files)} .adoc files")
        print()
        
        total_chunks = 0
        for file_path in adoc_files[:5]:  # Show first 5 files
            chunks = processor.process_file(file_path)
            total_chunks += len(chunks)
            print(f"\n📄 {Path(file_path).name}")
            print(f"   Chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"   Chunk {i+1}: {chunk.command_name} > {chunk.section}")
                print(f"      Preview: {chunk.content[:100]}...")
        
        print()
        print("=" * 60)
        print(f"Total files: {len(adoc_files)}")
        print(f"Sample chunks processed: {total_chunks}")
        print("Run without --dry-run to rebuild the index")
        
    else:
        print("🔄 Rebuilding index...")
        print("=" * 60)
        
        # Rebuild the index with progress
        stats = processor.rebuild_index(adoc_path, verbose=True)
        
        print()
        print("=" * 60)
        print("✅ Index rebuild complete!")
        print(f"   Files processed: {stats.get('files_processed', 0)}")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Errors: {stats.get('errors', 0)}")


if __name__ == "__main__":
    main()
