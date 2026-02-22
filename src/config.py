"""Configuration module for Router Config RAG Assistant.

Loads configuration from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # ──── Slack (Socket Mode) ────
    # Leave empty to disable Slack bot
    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN', '')
    SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN', '')

    # ──── Ollama ────
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    MODEL_NAME = os.getenv('MODEL_NAME', 'mistral:7b')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')

    # ──── Paths ────
    CHROMADB_PATH = os.getenv('CHROMADB_PATH', './data/chromadb')
    ADOC_FILES_PATH = os.getenv('ADOC_FILES_PATH', './data/adoc_files')

    # ──── RAG Tuning ────
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 5))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.4))
    MAX_THREAD_CONTEXT = int(os.getenv('MAX_THREAD_CONTEXT', 3))  # Q&A pairs

    # ──── Performance ────
    WORKER_THREADS = int(os.getenv('WORKER_THREADS', 3))
    QUERY_TIMEOUT = int(os.getenv('QUERY_TIMEOUT', 45))  # seconds
    RATE_LIMIT_SECONDS = int(os.getenv('RATE_LIMIT_SECONDS', 10))

    @classmethod
    def is_slack_enabled(cls) -> bool:
        """Check if Slack bot is enabled via config."""
        return bool(cls.SLACK_BOT_TOKEN and cls.SLACK_APP_TOKEN)

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration at startup.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        # Slack validation (only if enabled)
        if cls.is_slack_enabled():
            if not cls.SLACK_BOT_TOKEN.startswith('xoxb-'):
                raise ValueError("SLACK_BOT_TOKEN must start with 'xoxb-'")
            if not cls.SLACK_APP_TOKEN.startswith('xapp-'):
                raise ValueError("SLACK_APP_TOKEN must start with 'xapp-'")

        # Path validation
        if not cls.CHROMADB_PATH:
            raise ValueError("CHROMADB_PATH cannot be empty")
        if not cls.ADOC_FILES_PATH:
            raise ValueError("ADOC_FILES_PATH cannot be empty")

        # Validate paths don't contain traversal sequences
        # Use realpath to resolve any .. or symlinks and check the resolved path
        try:
            real_chromadb = os.path.realpath(cls.CHROMADB_PATH)
            real_adoc = os.path.realpath(cls.ADOC_FILES_PATH)
        except OSError as e:
            raise ValueError(f"Invalid path: {e}")
        
        if os.path.isabs(real_chromadb):
            raise ValueError("CHROMADB_PATH must be a relative path")
        if os.path.isabs(real_adoc):
            raise ValueError("ADOC_FILES_PATH must be a relative path")
        if '..' in real_chromadb or '..' in real_adoc:
            raise ValueError("Paths cannot contain '..' traversal sequences")

        # Ollama validation
        if not cls.MODEL_NAME:
            raise ValueError("MODEL_NAME cannot be empty")
        if not cls.EMBEDDING_MODEL:
            raise ValueError("EMBEDDING_MODEL cannot be empty")

    @classmethod
    def get_chromadb_path(cls) -> Path:
        """Get absolute path to ChromaDB directory."""
        return Path(cls.CHROMADB_PATH).resolve()

    @classmethod
    def get_adoc_files_path(cls) -> Path:
        """Get absolute path to ADOC files directory."""
        return Path(cls.ADOC_FILES_PATH).resolve()

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        cls.get_chromadb_path().mkdir(parents=True, exist_ok=True)
        cls.get_adoc_files_path().mkdir(parents=True, exist_ok=True)
