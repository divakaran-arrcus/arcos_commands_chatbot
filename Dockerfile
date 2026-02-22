# Router Config RAG Assistant - Docker Image
# Multi-service image for Slack bot and Chainlit web UI

FROM python:3.11-slim

# Labels for container metadata
LABEL maintainer="Router Config Assistant"
LABEL description="RAG assistant for router CLI commands"
LABEL version="1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY chainlit.md ./chainlit.md
COPY .chainlit/ ./.chainlit/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check - verify Python can import modules and Ollama is reachable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.config import Config; import socket; s=socket.socket(); s.settimeout(2); s.connect((Config.OLLAMA_HOST.split('//')[1].split(':')[0], int(Config.OLLAMA_HOST.split(':')[-1]))); s.close(); print('ok')" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "src/slack_bot.py"]
