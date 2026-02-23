# Router Config RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about router CLI commands and configurations from AsciiDoc reference documentation. Uses a local LLM (Ollama) running on-premise with no external data access.

## Features

- 🔍 **Semantic Search**: Find relevant CLI commands and configurations using natural language
- 💬 **Dual Interface**: Slack bot + ChatGPT-style web UI (Chainlit)
- 🔒 **On-Premise**: All processing happens locally - no data leaves your infrastructure
- 📚 **Source Citations**: Every answer includes references to source documentation
- 🧵 **Thread Context**: Remembers conversation history for follow-up questions
- ⚡ **Concurrent Processing**: Handles multiple queries with thread pool workers

## Architecture

```
┌───────────────┐     ┌─────────────────────────────────────────────────┐
│  Slack User   │◄───►│  Slack Bot (Socket Mode)                       │
└───────────────┘     │       │                                        │
                      │       ▼                                        │
┌───────────────┐     │  ┌──────────────────────────┐                  │
│  Web Browser  │◄───►│  │     RAG Engine            │                 │
│  (Chainlit)   │     │  │  ┌─────────┐ ┌─────────┐ │                 │
└───────────────┘     │  │  │ChromaDB │ │ Ollama  │ │                 │
                      │  │  │(vector) │ │ (LLM)   │ │                 │
                      │  │  └─────────┘ └─────────┘ │                 │
                      │  └──────────────────────────┘                  │
                      └─────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.11+** (for local development)
- **Docker & Docker Compose** (for deployment)
- **Ollama** (runs on host for best CPU performance)
- **Slack App** (optional, for Slack integration)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd arcos_commands_chatbot

# Create virtual environment and install dependencies
make install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### 2. Install and Start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull required models
ollama pull mistral:7b           # LLM for answer generation
ollama pull nomic-embed-text     # Embedding model for semantic search
```

### 3. Index Documentation

Place your `.adoc` CLI reference files in `data/adoc_files/`, then:

```bash
# Rebuild the vector index
make rebuild-index

# Or preview chunks without storing
make rebuild-index DRY_RUN=1
```

### 4. Run the Application

**Development (local):**
```bash
# Start web UI only
make web

# Start Slack bot (requires Slack tokens in .env)
make slack
```

**Production (Docker):**
```bash
# Build and start all services
make build
make start

# View logs
make logs
```

Access the web UI at http://localhost:8080

## Configuration

Configuration is managed via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SLACK_BOT_TOKEN` | Slack bot token (xoxb-...) | - |
| `SLACK_APP_TOKEN` | Slack app-level token (xapp-...) | - |
| `OLLAMA_HOST` | Ollama API endpoint | `http://localhost:11434` |
| `MODEL_NAME` | LLM model for generation | `mistral:7b` |
| `EMBEDDING_MODEL` | Model for embeddings | `nomic-embed-text` |
| `CHROMADB_PATH` | Vector database path | `./data/chromadb` |
| `ADOC_FILES_PATH` | AsciiDoc source files path | `./data/adoc_files` |
| `WORKER_THREADS` | Concurrent query workers | `3` |
| `RATE_LIMIT_SECONDS` | Per-user rate limit | `10` |

## Slack Bot Setup

### Requirements
- Slack workspace admin access (or request from admin)

### Steps

1. Go to https://api.slack.com/apps → "Create New App" → "From scratch"
2. App Name: **Router Config Assistant**
3. Enable **Socket Mode** → Generate app-level token with `connections:write` scope
4. Add **Bot Token Scopes**:
   - `chat:write`
   - `app_mentions:read`
   - `channels:history`
   - `im:history`
   - `channels:read`
5. Subscribe to **Events**:
   - `app_mention`
   - `message.im`
6. Enable "Messages Tab" in **App Home**
7. Install app to workspace
8. Copy tokens to `.env`:
   - Bot User OAuth Token (`xoxb-`) → `SLACK_BOT_TOKEN`
   - App-Level Token (`xapp-`) → `SLACK_APP_TOKEN`
9. Invite bot to channel: `/invite @Router Config Assistant`

## Project Structure

```
arcos_commands_chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # AsciiDoc parsing and chunking
│   ├── rag_engine.py          # RAG pipeline (search + generate)
│   ├── slack_bot.py           # Slack bot implementation
│   └── web_ui.py              # Chainlit web interface
├── scripts/
│   └── rebuild_index.py       # Index rebuild utility
├── data/
│   ├── adoc_files/            # Place .adoc files here
│   └── chromadb/              # Vector database (generated)
├── .chainlit/                 # Chainlit configuration
├── docker-compose.yml         # Docker deployment
├── Dockerfile                 # Container image
├── Makefile                   # Development commands
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## Available Commands

Run `make help` to see all available commands:

```
Development:
  make install        Install Python dependencies
  make test-index     Test document processor
  make test-rag       Test RAG engine with sample query
  make web            Start Chainlit web UI (dev mode)
  make slack          Start Slack bot (dev mode)

Docker Operations:
  make build          Build Docker images
  make start          Start all services (detached)
  make stop           Stop all services
  make restart        Restart all services
  make logs           Tail all logs
  make logs-slack     Tail Slack bot logs
  make logs-web       Tail web UI logs

Maintenance:
  make rebuild-index  Rebuild ChromaDB index
  make backup         Backup ChromaDB index
  make clean          Clean up generated files
  make ollama-status  Check Ollama status
```

## Model Selection

| Model | Quality | Speed (32-core CPU) | Context | Recommendation |
|-------|---------|---------------------|---------|----------------|
| `mistral:7b` | ★★★★☆ | ~5-8s first token | 32K | **Primary choice** |
| `gemma2:2b` | ★★★☆☆ | ~2-4s first token | 8K | Faster, lower quality |
| `llama3:8b` | ★★★★★ | ~8-12s first token | 8K | Highest quality |

To change the model, update `MODEL_NAME` in `.env` and ensure it's pulled in Ollama.

## Troubleshooting

### Ollama not responding
```bash
# Check Ollama status
make ollama-status

# Restart Ollama
sudo systemctl restart ollama
```

### Module import errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Or run with venv Python directly
./venv/bin/python src/slack_bot.py
```

### Slack bot not connecting
- Verify tokens start with correct prefixes (`xoxb-` and `xapp-`)
- Check Socket Mode is enabled in Slack app settings
- Ensure app is installed to workspace

### Slow responses
- Try a smaller model (`gemma2:2b`)
- Reduce `TOP_K_RESULTS` in `.env`
- Increase `WORKER_THREADS` for concurrent requests

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
