"""RAG (Retrieval-Augmented Generation) Engine for Router CLI Documentation.

This module handles semantic search over indexed documentation and LLM-based
answer generation using Ollama.
"""
import logging
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import chromadb
import ollama

from src.config import Config

logger = logging.getLogger(__name__)

# System prompt for the LLM
SYSTEM_PROMPT = """You are a router CLI reference assistant. You answer questions about
router commands, syntax, parameters, and configuration examples based SOLELY on the
provided CLI reference documentation.

RULES — FOLLOW STRICTLY:
1. Answer ONLY from the provided documentation. Never invent commands or parameters.
2. If the answer is NOT in the provided context, say: "I don't have information about
   that in the CLI reference files. Could you try asking about a specific command?"
3. ALWAYS cite the source file: [Source: filename.adoc]
4. When showing command syntax, use exact syntax from the documentation.
5. Include parameter descriptions when available.
6. Be precise and technical — your audience is network engineers.
7. Use code blocks for command examples.

RESPONSE FORMAT:
- Direct answer first
- Command syntax in a code block
- Parameter explanations if relevant
- Usage examples if available in the docs
- Source citation(s) at the end
"""


class RAGEngine:
    """Retrieval-Augmented Generation engine for router CLI docs."""

    def __init__(self,
                 chromadb_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 ollama_host: Optional[str] = None):
        """Initialize the RAG engine.

        Args:
            chromadb_path: Path to ChromaDB persistence directory.
            model_name: Ollama model name for generation.
            embedding_model: Ollama embedding model name.
            ollama_host: Ollama API host URL.
        """
        # Use config defaults if not provided
        chromadb_path = chromadb_path or Config.CHROMADB_PATH
        model_name = model_name or Config.MODEL_NAME
        embedding_model = embedding_model or Config.EMBEDDING_MODEL
        ollama_host = ollama_host or Config.OLLAMA_HOST

        # Initialize ChromaDB client (read-only for queries)
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_collection("router_cli_docs")

        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host

        # Initialize Ollama client
        self._ollama_client = None

        # Worker pool for concurrent queries
        self.executor = ThreadPoolExecutor(max_workers=Config.WORKER_THREADS)

    @property
    def ollama_client(self):
        """Lazy initialization of Ollama client."""
        if self._ollama_client is None:
            self._ollama_client = ollama.Client(host=self.ollama_host)
        return self._ollama_client

    def preprocess_query(self, raw_query: str) -> str:
        """Clean Slack formatting and normalize query text.

        Args:
            raw_query: Raw query from user.

        Returns:
            Cleaned query string.
        """
        # Remove <@USER_ID> mentions
        query = re.sub(r'<@U\w+>', '', raw_query)

        # Remove emoji shortcodes :emoji:
        query = re.sub(r':\w+:', '', query)

        # Remove URLs
        query = re.sub(r'http[s]?://\S+', '', query)

        # Remove Slack channel references
        query = re.sub(r'<#\w+\|?[^>]*>', '', query)

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        return query

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.4,
               protocol_filter: Optional[str] = None) -> list[dict]:
        """Semantic search over indexed documentation.

        Args:
            query: User query.
            top_k: Number of results to retrieve.
            score_threshold: Minimum similarity score (0-1).
            protocol_filter: Optional protocol filter (e.g., 'bgp', 'ospf').

        Returns:
            List of retrieved chunks with metadata.
        """
        # Generate query embedding
        response = self.ollama_client.embeddings(
            model=self.embedding_model,
            prompt=query
        )
        query_embedding = response['embedding']

        # Build filter if protocol specified
        where_filter = None
        if protocol_filter:
            where_filter = {"protocol": protocol_filter}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Process and filter results
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= score_threshold:
                    chunks.append({
                        'content': doc,
                        'source_file': results['metadatas'][0][i].get('source_file', ''),
                        'command_name': results['metadatas'][0][i].get('command_name', ''),
                        'section': results['metadatas'][0][i].get('section', ''),
                        'protocol': results['metadatas'][0][i].get('protocol', ''),
                        'chunk_type': results['metadatas'][0][i].get('chunk_type', ''),
                        'heading_chain': results['metadatas'][0][i].get('heading_chain', ''),
                        'similarity': similarity
                    })

        logger.info(f"Search for '{query}' returned {len(chunks)} relevant chunks")
        return chunks

    def build_prompt(self, question: str, chunks: list[dict],
                    thread_context: Optional[list[dict]] = None) -> str:
        """Construct the full prompt for the LLM.

        Args:
            question: User question.
            chunks: Retrieved document chunks.
            thread_context: Optional conversation history.

        Returns:
            Formatted prompt string.
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source_file', 'Unknown')
            section = chunk.get('section', '')
            heading = chunk.get('heading_chain', '')

            header = f"--- Context {i} (Source: {source}"
            if section:
                header += f", Section: {section}"
            header += ") ---"

            context_parts.append(f"{header}\n{chunk['content']}")

        context = "\n\n".join(context_parts)

        # Build conversation context if available
        conversation = ""
        if thread_context:
            conversation_parts = []
            for msg in thread_context[-6:]:  # Last 3 Q&A pairs
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_parts.append(f"{role.upper()}: {content}")
            conversation = "\n".join(conversation_parts)

        # Build final prompt
        if conversation:
            prompt = f"""Previous conversation:
{conversation}

Current question: {question}

Relevant documentation:
{context}

Based on the documentation above, please answer the question. If the documentation
doesn't contain the answer, say so clearly and cite your sources.
"""
        else:
            prompt = f"""Question: {question}

Relevant documentation:
{context}

Based on the documentation above, please answer the question. If the documentation
doesn't contain the answer, say so clearly and cite your sources.
"""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """Call Ollama and return the response text.

        Args:
            prompt: Formatted prompt for the LLM.

        Returns:
            Generated answer text.
        """
        try:
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.1,
                    "num_ctx": 8192
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def format_for_slack(self, answer: str, chunks: list[dict]) -> str:
        """Format response with Slack markdown and source citations.

        Args:
            answer: Generated answer.
            chunks: Retrieved chunks for citation.

        Returns:
            Formatted response.
        """
        # Extract unique source files
        sources = set()
        for chunk in chunks:
            if chunk.get('source_file'):
                sources.add(chunk['source_file'])

        # Add sources if not already in answer
        if sources:
            source_list = ", ".join(sorted(sources))
            if source_list not in answer:
                answer += f"\n\n📚 *Sources:* {source_list}"

        return answer

    def answer_query(self, question: str,
                    thread_context: Optional[list[dict]] = None) -> str:
        """Full RAG pipeline: preprocess → search → prompt → generate → format.

        Args:
            question: User question.
            thread_context: Optional conversation history.

        Returns:
            Formatted answer string.
        """
        # Preprocess query
        clean_query = self.preprocess_query(question)

        # Search for relevant chunks
        chunks = self.search(
            clean_query,
            top_k=Config.TOP_K_RESULTS,
            score_threshold=Config.SIMILARITY_THRESHOLD
        )

        # Handle no results
        if not chunks:
            return ("I couldn't find relevant information in the CLI reference "
                    "documentation. Could you rephrase or ask about a specific command?")

        # Build prompt
        prompt = self.build_prompt(clean_query, chunks, thread_context)

        # Generate answer
        answer = self.generate_answer(prompt)

        # Format for output
        formatted = self.format_for_slack(answer, chunks)

        return formatted


def main():
    """CLI entry point for testing the RAG engine."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Test the RAG engine')
    parser.add_argument('query', nargs='+', help='Query to search for')
    args = parser.parse_args()

    query = ' '.join(args.query)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize engine
    engine = RAGEngine()

    # Run query
    print(f"\n🔍 Query: {query}\n")
    start = time.time()
    answer = engine.answer_query(query)
    elapsed = time.time() - start

    print(f"Answer ({elapsed:.1f}s):\n{answer}\n")


if __name__ == "__main__":
    main()
