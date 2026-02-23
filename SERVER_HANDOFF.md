# Server Handoff Document

## Current Status

The Router Config RAG Assistant is deployed on the Linux server with Docker. The index rebuild was successful, but the chatbot is returning "I couldn't find relevant information" for all queries.

## Issue to Debug

**Problem:** Search returns no results even though ChromaDB has indexed data.

**Possible Causes:**
1. ChromaDB collection is empty or data wasn't persisted
2. Similarity threshold (0.4) is too high
3. Embedding model mismatch between indexing and querying
4. Distance metric issue in ChromaDB

## Debugging Steps

### Step 1: Check ChromaDB Collection

Run on the server:
```bash
make test-rag
```

This will show:
- Collection count (should be > 0)
- Search results with similarity scores

### Step 2: Check Docker Logs

```bash
make logs-web
```

Look for lines like:
```
Search for 'query' returned X relevant chunks (total results: Y)
```

### Step 3: Verify Data Persistence

```bash
# Check if ChromaDB directory has data
ls -la data/chromadb/

# Should see collections directory and other files
```

### Step 4: Test Search Directly in Docker

```bash
docker compose exec web-ui python -c "
from src.rag_engine import RAGEngine
engine = RAGEngine()
print(f'Collection count: {engine.collection.count()}')
chunks = engine.search('bgp', top_k=5, score_threshold=0.0)
print(f'Found {len(chunks)} chunks')
for c in chunks[:3]:
    print(f'  - {c[\"source_file\"]}: similarity={c[\"similarity\"]:.4f}')
"
```

## Quick Fixes to Try

### Lower Similarity Threshold

Edit `.env`:
```bash
SIMILARITY_THRESHOLD=0.1
```

Then restart:
```bash
make restart
```

### Rebuild Index with Verbose Output

```bash
make rebuild-index-docker
```

Watch for:
- Number of files processed
- Number of chunks stored
- Any errors during embedding generation

## Files Modified During Session

| File | Change |
|------|--------|
| `src/web_ui.py` | Fixed module import error |
| `src/slack_bot.py` | Created Slack bot implementation |
| `src/config.py` | Allow absolute paths for Docker |
| `docker-compose.yml` | Fixed ChromaDB read-only issue |
| `Makefile` | Added `rebuild-index-docker` command |
| `scripts/rebuild_index.py` | Added verbose output and Ollama connection test |
| `src/document_processor.py` | Added progress output |
| `src/rag_engine.py` | Added debug logging |
| `README.md` | Updated documentation |

## Architecture

```
┌───────────────┐     ┌─────────────────────────────────────────────────┐
│  Web Browser  │◄───►│  Docker Container (web-ui)                      │
│  :8080        │     │  - Chainlit                                     │
└───────────────┘     │  - RAG Engine                                    │
                      │  - ChromaDB (mounted volume)                     │
                      └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                      ┌─────────────────────────────────────────────────┐
                      │  Host Machine                                    │
                      │  - Ollama (:11434)                               │
                      │  - data/chromadb/ (persisted)                    │
                      │  - data/adoc_files/ (89 .adoc files)             │
                      └─────────────────────────────────────────────────┘
```

## Next Steps After Debugging

1. Once search works, test with real queries
2. Configure Slack tokens when ready
3. Consider switching to `llama3.1:8b` for better quality (already pulled)

## Useful Commands

```bash
# Check Ollama status
ollama list

# Rebuild index
make rebuild-index-docker

# View logs
make logs

# Restart services
make restart

# Stop everything
make stop
```
