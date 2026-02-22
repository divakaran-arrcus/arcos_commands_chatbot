"""Document Processor for AsciiDoc CLI Reference Files.

This module handles parsing, chunking, and indexing of AsciiDoc CLI reference
documentation into ChromaDB for semantic search.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import chromadb
import ollama

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of documentation with metadata."""
    content: str
    source_file: str
    command_name: str
    section: str
    protocol: Optional[str] = None
    chunk_type: Optional[str] = None  # syntax, parameters, examples, description
    heading_chain: str = ""  # e.g., "show bgp > Parameters"


class AdocParser:
    """Parse AsciiDoc CLI reference files into structured sections."""

    # AsciiDoc heading pattern: = Title, == Subtitle, === Sub-subtitle
    HEADING_RE = re.compile(r'^(={1,4})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_RE = re.compile(r'^(----|\.\.\.\.)$', re.MULTILINE)
    TABLE_DELIM_RE = re.compile(r'^\|===$', re.MULTILINE)

    # Protocol keywords for auto-detection
    PROTOCOLS = ['bgp', 'isis', 'ospf', 'mpls', 'ldp', 'rsvp', 'vrf', 'vlan',
                 'acl', 'nat', 'qos', 'bfd', 'pim', 'evpn', 'ldp', 'vrrp']

    def parse_file(self, file_path: str) -> list[DocumentChunk]:
        """Parse an .adoc file into chunks with metadata."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Get filename for source tracking
        filename = os.path.basename(file_path)

        # Extract command name from top-level heading
        command_name = self._extract_command_name(content)

        # Split into sections by heading hierarchy
        sections = self._split_by_headings(content, filename)

        # Convert sections to chunks
        chunks = []
        for section in sections:
            # Detect protocol
            protocol = self._detect_protocol(section['content'])

            # Classify section type
            chunk_type = self._classify_section(section['heading'])

            # Build heading chain
            heading_chain = section.get('parent_heading', command_name)
            if section['heading']:
                heading_chain += f" > {section['heading']}"

            chunk = DocumentChunk(
                content=section['content'],
                source_file=filename,
                command_name=command_name,
                section=section['heading'],
                protocol=protocol,
                chunk_type=chunk_type,
                heading_chain=heading_chain
            )
            chunks.append(chunk)

        return chunks

    def _extract_command_name(self, content: str) -> str:
        """Extract command name from top-level heading."""
        match = re.search(r'^=\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Unknown Command"

    def _split_by_headings(self, content: str, filename: str) -> list[dict]:
        """Split content by heading levels, preserving hierarchy."""
        sections = []
        lines = content.split('\n')

        current_section = {
            'heading': '',
            'parent_heading': '',
            'content': '',
            'level': 0
        }

        for i, line in enumerate(lines):
            # Check for heading
            heading_match = self.HEADING_RE.match(line)
            if heading_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    # Split large sections into smaller chunks
                    large_sections = self._split_large_section(current_section)
                    sections.extend(large_sections)

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Update parent heading based on level
                if level == 1:
                    current_section = {
                        'heading': '',
                        'parent_heading': heading_text,
                        'content': '',
                        'level': level
                    }
                else:
                    current_section = {
                        'heading': heading_text,
                        'parent_heading': current_section.get('parent_heading', ''),
                        'content': '',
                        'level': level
                    }
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content'].strip():
            large_sections = self._split_large_section(current_section)
            sections.extend(large_sections)

        return sections

    def _split_large_section(self, section: dict) -> list[dict]:
        """Split a large section into smaller chunks."""
        max_chunk_size = 2000  # characters
        content = section['content'].strip()
        
        if len(content) <= max_chunk_size:
            return [section]
        
        # Split by paragraphs (double newline)
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = {
            'heading': section['heading'],
            'parent_heading': section['parent_heading'],
            'content': '',
            'level': section['level']
        }
        
        for para in paragraphs:
            if len(current_chunk['content']) + len(para) + 2 > max_chunk_size:
                if current_chunk['content'].strip():
                    chunks.append(current_chunk)
                current_chunk = {
                    'heading': section['heading'],
                    'parent_heading': section['parent_heading'],
                    'content': '',
                    'level': section['level']
                }
            current_chunk['content'] += para + '\n\n'
        
        if current_chunk['content'].strip():
            chunks.append(current_chunk)
        
        return chunks if chunks else [section]

    def _detect_protocol(self, text: str) -> Optional[str]:
        """Auto-detect protocol from content keywords."""
        text_lower = text.lower()
        for protocol in self.PROTOCOLS:
            # Look for protocol as a whole word
            if re.search(r'\b' + protocol + r'\b', text_lower):
                return protocol
        return None

    def _classify_section(self, heading: str) -> Optional[str]:
        """Classify section type from heading text."""
        heading_lower = heading.lower()
        if 'syntax' in heading_lower:
            return 'syntax'
        elif 'parameter' in heading_lower:
            return 'parameters'
        elif 'example' in heading_lower:
            return 'examples'
        elif 'description' in heading_lower:
            return 'description'
        elif 'related' in heading_lower or 'see also' in heading_lower:
            return 'related'
        return None


class DocumentProcessor:
    """Process .adoc files into ChromaDB vector store."""

    def __init__(self,
                 chromadb_path: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 ollama_host: Optional[str] = None):
        """Initialize the document processor.

        Args:
            chromadb_path: Path to ChromaDB persistence directory.
            embedding_model: Ollama embedding model name.
            ollama_host: Ollama API host URL.
        """
        self.parser = AdocParser()

        # Use config defaults if not provided
        chromadb_path = chromadb_path or Config.CHROMADB_PATH
        embedding_model = embedding_model or Config.EMBEDDING_MODEL
        ollama_host = ollama_host or Config.OLLAMA_HOST

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(
            name="router_cli_docs",
            metadata={"hnsw:space": "cosine"}
        )

        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self._ollama_client = None

    @property
    def ollama_client(self):
        """Lazy initialization of Ollama client."""
        if self._ollama_client is None:
            self._ollama_client = ollama.Client(host=self.ollama_host)
        return self._ollama_client

    def load_adoc_files(self, directory: str) -> list[str]:
        """Discover all .adoc files in directory.

        Args:
            directory: Path to directory containing .adoc files.

        Returns:
            List of file paths.
        """
        adoc_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.adoc'):
                    adoc_files.append(os.path.join(root, file))
        return sorted(adoc_files)

    def process_file(self, file_path: str) -> list[DocumentChunk]:
        """Parse and chunk a single .adoc file.

        Args:
            file_path: Path to .adoc file.

        Returns:
            List of DocumentChunk objects.
        """
        return self.parser.parse_file(file_path)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text chunk using Ollama.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Truncate text if too long for embedding model (typical limit is ~8K tokens)
        # We'll limit to ~4000 characters as a safe upper bound
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length]
        
        response = self.ollama_client.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response['embedding']

    def store_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store chunks with embeddings in ChromaDB.

        Args:
            chunks: List of DocumentChunk objects to store.
        """
        if not chunks:
            return

        # Generate embeddings in batch
        embeddings = []
        texts = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk.content)
            embeddings.append(embedding)
            texts.append(chunk.content)
            metadatas.append({
                'source_file': chunk.source_file,
                'command_name': chunk.command_name,
                'section': chunk.section,
                'protocol': chunk.protocol or '',
                'chunk_type': chunk.chunk_type or '',
                'heading_chain': chunk.heading_chain
            })
            ids.append(f"{chunk.source_file}_{i}")

        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def rebuild_index(self, adoc_directory: str, dry_run: bool = False) -> dict:
        """Full index rebuild: parse all files, chunk, embed, store.

        Args:
            adoc_directory: Path to directory containing .adoc files.
            dry_run: If True, parse and chunk but don't store.

        Returns:
            Statistics dictionary.
        """
        logger.info(f"Starting index rebuild from {adoc_directory}")

        # Discover files
        adoc_files = self.load_adoc_files(adoc_directory)
        logger.info(f"Found {len(adoc_files)} .adoc files")

        # Clear existing collection if not dry run
        if not dry_run:
            try:
                self.client.delete_collection("router_cli_docs")
                self.collection = self.client.get_or_create_collection(
                    name="router_cli_docs",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass  # Collection might not exist

        # Process all files
        all_chunks = []
        stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0
        }

        for file_path in adoc_files:
            logger.info(f"Processing: {file_path}")
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
            stats['files_processed'] += 1
            stats['total_chunks'] += len(chunks)

        # Store chunks (unless dry run)
        if not dry_run:
            logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB...")
            self.store_chunks(all_chunks)
        else:
            logger.info(f"Dry run: would store {len(all_chunks)} chunks")

        # Calculate statistics
        if all_chunks:
            total_chars = sum(len(c.content) for c in all_chunks)
            stats['avg_chunk_size'] = total_chars / len(all_chunks)

        logger.info(f"Index rebuild complete: {stats}")
        return stats

    def get_stats(self) -> dict:
        """Return index statistics."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': 'router_cli_docs'
        }


def main():
    """CLI entry point for document processor."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process AsciiDoc CLI reference files into ChromaDB'
    )
    parser.add_argument(
        '--adoc-dir',
        default=Config.ADOC_FILES_PATH,
        help='Directory containing .adoc files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview chunks without storing'
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Ensure directories exist
    Config.ensure_directories()

    # Run processor
    processor = DocumentProcessor()
    stats = processor.rebuild_index(args.adoc_dir, dry_run=args.dry_run)

    print(f"\n✅ Index rebuild complete!")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Avg chunk size: {stats['avg_chunk_size']:.0f} chars")


if __name__ == "__main__":
    main()
