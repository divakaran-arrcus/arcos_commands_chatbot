# Router Configuration RAG Assistant — Enhanced Project Plan

## Project Overview

A Retrieval-Augmented Generation (RAG) assistant for Slack that answers questions about router CLI commands and configurations from AsciiDoc (`.adoc`) reference documentation, using a local LLM (Ollama) running on-premise with no external data access.

### Requirements Summary

| Requirement | Detail |
|---|---|
| **Documentation** | 89 `.adoc` CLI reference files (~3MB) in `Command_Line_Interface/` |
| **Users** | 100 total, ~5 concurrent |
| **Infrastructure** | On-premise Debian 12 VM — 32 vCPUs (Xeon Skylake @ 2.1GHz), 64GB RAM, ~500GB disk |
| **GPU** | None — CPU-only inference |
| **LLM** | Ollama (local, CPU) |
| **Interface** | Slack bot (single channel) + Chainlit web UI (ChatGPT-style) |
| **Update Frequency** | Manual, once per release |
| **Target Response Time** | ≤10 seconds (realistic for CPU) |
| **Key Features** | Thread context, source citations, hallucination prevention |

---

## Architecture

### High-Level Design

```
┌───────────────┐     ┌─────────────────────────────────────────────────┐
│  Slack User   │◄───►│  Slack Bot (Socket Mode)                       │
└───────────────┘     │       │                                        │
                      │       ▼                                        │
┌───────────────┐     │  ┌──────────────────────────┐                  │
│  Web Browser  │◄───►│  │     RAG Engine            │                 │
│  (Chainlit)   │     │  │  ┌─────────┐ ┌─────────┐ │                 │
└───────────────┘     │  │  │ChromaDB │ │ Ollama  │ │                 │
                      │  │  │(read)   │ │ (LLM)   │ │                 │
                      │  │  └─────────┘ └─────────┘ │                 │
                      │  └──────────────────────────┘                  │
                      └─────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  Offline Pipeline (run manually per release)     │
│                                                  │
│  Command_Line_Interface/*.adoc                   │
│        │                                         │
│        ▼                                         │
│  Document Processor (parse + chunk + embed)      │
│        │                                         │
│        ▼                                         │
│  ChromaDB (persistent vector store)              │
└──────────────────────────────────────────────────┘
```

### Why No Message Queue?

The original plan included RabbitMQ. We removed it because:

- **~5 concurrent users** — a `ThreadPoolExecutor(max_workers=3)` handles this trivially
- Removes an entire service to deploy, monitor, and debug
- Eliminates network hops and serialization overhead (saves ~200-500ms per query)
- If scaling is needed later, RabbitMQ can be added back as a drop-in layer

### Components

| Component | Purpose | Technology |
|---|---|---|
| **Slack Bot** | Receives questions via Socket Mode, dispatches to workers, sends responses | `slack-bolt` (Python) |
| **Web UI** | ChatGPT-style browser interface with streaming, history, citations | Chainlit |
| **Async Worker Pool** | Processes queries concurrently (3 threads) | `concurrent.futures.ThreadPoolExecutor` |
| **RAG Engine** | Semantic search + LLM generation (shared by Slack + Web UI) | Custom Python |
| **Document Processor** | One-time indexing of `.adoc` CLI reference files | Custom Python |
| **Vector Store** | Embeddings storage and similarity search | ChromaDB (embedded) |
| **LLM** | Local inference for answer generation | Ollama (`mistral:7b`) |
| **Embedding Model** | Text embedding for semantic search | Ollama (`EMBEDDING_MODEL`, default `nomic-embed-text`) |

---

## Technology Stack

### LLM Model Selection

> [!IMPORTANT]
> With 32 cores and 64GB RAM, you can comfortably run **`mistral:7b`** — no need to compromise on the smaller `gemma2:2b`.

| Model | Quality | Speed (est. 32-core CPU) | Context Window | Recommendation |
|---|---|---|---|---|
| `mistral:7b` | ★★★★☆ | ~5-8s first token, 6-12s full response | 32K tokens | **Primary choice** |
| `gemma2:2b` | ★★★☆☆ | ~2-4s first token, 3-6s full response | 8K tokens | Fallback if speed matters more than quality |
| `llama3:8b` | ★★★★★ | ~8-12s first token, 10-15s full response | 8K tokens | Alternative if quality is paramount |
| `phi3:3.8b` | ★★★★☆ | ~3-5s first token, 5-8s full response | 128K tokens | Alternative with huge context window |

**Recommendation**: Start with `mistral:7b`. Its 32K context window means you can fit more retrieved chunks + thread context without truncation. If response time exceeds 12s, fall back to `gemma2:2b`.

### Vector Database

**ChromaDB** (embedded in Python)
- No separate service needed — runs in-process
- Persistent storage to disk
- Perfect for 89 documents (~3MB)
- Read-only during serving; writes only during index rebuild

### Embedding Model

**Default: `nomic-embed-text`** via Ollama (configurable via `EMBEDDING_MODEL` in `.env`)
- 768-dimensional embeddings
- Consistent with the Ollama stack (single dependency)
- Strong performance on technical content

> [!TIP]
> Also benchmark `mxbai-embed-large` via Ollama — it can outperform `nomic-embed-text` on technical retrieval tasks. Easy to swap by updating `EMBEDDING_MODEL`.

### Dependencies

```txt
# requirements.txt
slack-bolt>=1.18.0
slack-sdk>=3.23.0
chromadb>=0.4.22
ollama>=0.1.6
chainlit>=1.1.0
python-dotenv>=1.0.0
pyyaml>=6.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

> [!NOTE]
> We intentionally **do not** use LangChain. For this project's scope (single retrieval source, one LLM, simple pipeline), raw Ollama + ChromaDB APIs are simpler and more transparent. LangChain adds abstraction without value here.

---

## Phase 1: Foundation & Core RAG (Week 1-2)

### Task 1.1: Environment Setup

**Checklist:**

- [ ] Create project directory structure (see [Project Structure](#project-structure))
- [ ] Set up Python virtual environment (Python 3.10+):
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Create `.env.example` with all configuration variables
- [ ] Create `.env` (gitignored) with actual values
- [ ] Set up Git repository with `.gitignore`
- [ ] Verify Python environment works: `python -c "import chromadb; import ollama; print('OK')"`

**Deliverable**: Working Python environment with all dependencies installed

---

### Task 1.2: Ollama Setup

**Checklist:**

- [ ] Install Ollama on the Debian VM:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- [ ] Start Ollama service:
  ```bash
  sudo systemctl enable ollama
  sudo systemctl start ollama
  ```
- [ ] Pull models:
  ```bash
  ollama pull mistral:7b
  ollama pull nomic-embed-text  # default; match EMBEDDING_MODEL if changed
  ```
- [ ] Verify API accessible:
  ```bash
  curl http://localhost:11434/api/tags
  ```
- [ ] Benchmark inference on your hardware:
  ```bash
  time ollama run mistral:7b "Explain BGP routing in 3 sentences" --verbose
  ```
- [ ] Record performance metrics:
  - Tokens/second
  - Time-to-first-token
  - Total response latency
- [ ] Set Ollama environment for optimal CPU performance:
  ```bash
  # /etc/systemd/system/ollama.service.d/override.conf
  [Service]
  Environment="OLLAMA_NUM_PARALLEL=3"
  Environment="OLLAMA_MAX_LOADED_MODELS=2"
  ```
  Then: `sudo systemctl daemon-reload && sudo systemctl restart ollama`

> [!NOTE]
> `OLLAMA_NUM_PARALLEL=3` matches our worker pool size. This lets Ollama handle 3 concurrent inference requests efficiently across your 32 cores.

**Deliverable**: Ollama running with `mistral:7b` and the configured embedding model (`EMBEDDING_MODEL`), benchmarked

---

### Task 1.3: Document Processing Pipeline

**File**: `src/document_processor.py`

#### Understanding the AsciiDoc CLI Reference Structure

Before building the processor, examine your `.adoc` files to understand their structure. CLI reference files typically follow this pattern:

```asciidoc
= Command Name
:description: Brief description

== Syntax
  command [options] <arguments>

== Parameters
|===
| Parameter | Description | Default
| --flag    | Does something | false
|===

== Examples
  router# show bgp summary
  ...

== Related Commands
  see-also-command
```

#### Chunking Strategy for CLI Reference Docs

> [!IMPORTANT]
> Generic "split every 800 characters" chunking **will destroy** CLI reference structure. We use **section-aware chunking** that preserves command boundaries.

**Strategy**:
1. **Primary split**: By top-level AsciiDoc heading (`= Command Name`) — each command becomes at minimum one document
2. **Secondary split**: By second-level heading (`== Syntax`, `== Parameters`, `== Examples`) within large commands
3. **Keep intact**: Never split in the middle of a code block, table, or parameter list
4. **Metadata enrichment**: Every chunk carries `command_name`, `section`, `source_file`, and `protocol` (if identifiable)
5. **Hierarchy context**: Each chunk's text is prefixed with its parent heading chain (e.g., `Command: show bgp > Section: Examples`)

**Checklist:**

- [ ] Create AsciiDoc parser that understands heading hierarchy (`=`, `==`, `===`)
- [ ] Implement section-aware chunking:
  - [ ] Split by `=` headings (one chunk per top-level command)
  - [ ] Sub-split large sections by `==` headings
  - [ ] Never break inside code blocks (delimited by `----` or `....`)
  - [ ] Never break inside tables (delimited by `|===`)
  - [ ] Target chunk size: 500-1500 characters (flex to preserve structure)
  - [ ] Overlap: include parent heading chain in each sub-chunk (provides context without duplicating full content)
- [ ] Extract metadata per chunk:
  - [ ] `source_file`: filename
  - [ ] `command_name`: top-level heading
  - [ ] `section`: current section heading  
  - [ ] `protocol`: auto-detect from content (BGP, ISIS, OSPF, MPLS, etc.)
  - [ ] `chunk_type`: syntax | parameters | examples | description
- [ ] Generate embeddings using the configured embedding model (`EMBEDDING_MODEL`) via Ollama
- [ ] Store in ChromaDB with metadata
- [ ] Create rebuild script: `scripts/rebuild_index.py`
- [ ] Add `--dry-run` flag to preview chunks without storing
- [ ] Test with 5-10 sample adoc files, validate chunk quality
- [ ] Log statistics: total chunks, avg chunk size, chunks per file

**Code Structure:**

```python
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import chromadb
import ollama

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
    
    def parse_file(self, file_path: str) -> list[dict]:
        """Parse an .adoc file into sections with hierarchy."""
        
    def _split_by_headings(self, content: str) -> list[dict]:
        """Split content by heading levels, preserving hierarchy."""
        
    def _detect_protocol(self, text: str) -> Optional[str]:
        """Auto-detect protocol from content keywords."""
        protocols = ['bgp', 'isis', 'ospf', 'mpls', 'ldp', 'rsvp', 
                     'vrf', 'vlan', 'acl', 'nat', 'qos', 'bfd', 'pim']
        # Match against content
        
    def _classify_section(self, heading: str) -> Optional[str]:
        """Classify section type from heading text."""
        # syntax, parameters, examples, description, related

class DocumentProcessor:
    """Process .adoc files into ChromaDB vector store."""
    
    def __init__(self, chromadb_path: str, embedding_model: str = "nomic-embed-text",
                 ollama_host: str = "http://localhost:11434"):
        self.parser = AdocParser()
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(
            name="router_cli_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
    
    def load_adoc_files(self, directory: str) -> list[str]:
        """Discover all .adoc files in directory."""
        
    def process_file(self, file_path: str) -> list[DocumentChunk]:
        """Parse and chunk a single .adoc file."""
        
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text chunk using Ollama."""
        response = ollama.embed(model=self.embedding_model, input=text)
        return response["embeddings"][0]
    
    def store_chunks(self, chunks: list[DocumentChunk]):
        """Store chunks with embeddings in ChromaDB."""
        
    def rebuild_index(self, adoc_directory: str, dry_run: bool = False):
        """Full index rebuild: parse all files, chunk, embed, store."""
        
    def get_stats(self) -> dict:
        """Return index statistics."""
```

**Deliverable**: All 89 `.adoc` files indexed in ChromaDB with structure-preserving chunks

---

### Task 1.4: RAG Query Engine

**File**: `src/rag_engine.py`

**Checklist:**

- [ ] Initialize ChromaDB client (read-only mode for queries)
- [ ] Implement query preprocessing:
  - [ ] Remove Slack formatting (`<@USER_ID>`, emoji shortcodes, URLs)
  - [ ] Normalize whitespace
  - [ ] Expand common abbreviations (e.g., "config" → "configuration")
- [ ] Implement semantic search:
  - [ ] Generate query embedding via the configured embedding model
  - [ ] Search ChromaDB for top-K similar chunks (K=5)
  - [ ] Filter by relevance score threshold (≥0.4)
  - [ ] Optionally filter by metadata (protocol, command_name)
- [ ] Build context-aware prompt:
  - [ ] System instructions (stay within docs, cite sources, no hallucinations)
  - [ ] Retrieved chunks with source metadata
  - [ ] Thread context (last 2-3 Q&A pairs, not 5 — to fit context window)
  - [ ] User question
- [ ] Call Ollama with built prompt
- [ ] Post-process response:
  - [ ] Validate citations are present
  - [ ] Format for Slack (code blocks, bullet lists)
  - [ ] Append source file references
- [ ] Handle edge cases:
  - [ ] No relevant documents found → polite "not found" message
  - [ ] Ollama timeout (>30s) → retry once, then error message
  - [ ] Empty or very short user query → ask for clarification

**Code Structure:**

```python
import ollama
import chromadb
import re
from typing import Optional

class RAGEngine:
    """Retrieval-Augmented Generation engine for router CLI docs."""
    
    def __init__(self, chromadb_path: str, model_name: str = "mistral:7b",
                 embedding_model: str = "nomic-embed-text",
                 ollama_host: str = "http://localhost:11434"):
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_collection("router_cli_docs")
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
    
    def preprocess_query(self, raw_query: str) -> str:
        """Clean Slack formatting and normalize query text."""
        # Remove <@USER_ID> mentions
        # Remove emoji shortcodes :emoji:
        # Strip extra whitespace
    
    def search(self, query: str, top_k: int = 5, 
               score_threshold: float = 0.4,
               protocol_filter: Optional[str] = None) -> list[dict]:
        """Semantic search over indexed documentation."""
        embedding = ollama.embed(model=self.embedding_model, input=query)
        
        where_filter = None
        if protocol_filter:
            where_filter = {"protocol": protocol_filter}
        
        results = self.collection.query(
            query_embeddings=[embedding["embeddings"][0]],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Filter by score threshold and return
    
    def build_prompt(self, question: str, chunks: list[dict],
                     thread_context: Optional[list[dict]] = None) -> str:
        """Construct the full prompt for the LLM."""
    
    def generate_answer(self, prompt: str) -> str:
        """Call Ollama and return the response text."""
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1, "num_ctx": 8192}
        )
        return response["message"]["content"]
    
    def format_for_slack(self, answer: str, chunks: list[dict]) -> str:
        """Format response with Slack markdown and source citations."""
    
    def answer_query(self, question: str,
                     thread_context: Optional[list[dict]] = None) -> str:
        """Full RAG pipeline: preprocess → search → prompt → generate → format."""
        clean_query = self.preprocess_query(question)
        chunks = self.search(clean_query)
        
        if not chunks:
            return ("I couldn't find relevant information in the CLI reference "
                    "documentation. Could you rephrase or ask about a specific command?")
        
        prompt = self.build_prompt(clean_query, chunks, thread_context)
        answer = self.generate_answer(prompt)
        return self.format_for_slack(answer, chunks)
```

**System Prompt:**

```python
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
```

**Deliverable**: Working RAG engine that retrieves CLI reference chunks and generates cited answers

---

## Phase 2: Slack Integration (Week 2)

### Task 2.1: Getting Slack App Admin Access

> [!IMPORTANT]
> You need a Slack workspace admin (or someone with permission to install apps) to create the Slack app. Here's how to handle this.

**Option A — Request Admin to Create the App for You:**

Provide this checklist to your Slack workspace admin:

1. Go to https://api.slack.com/apps → "Create New App" → "From scratch"
2. App Name: **Router Config Assistant**
3. Select your workspace
4. Go to **Socket Mode** → Enable Socket Mode → Generate an **App-Level Token** with `connections:write` scope → save token (starts with `xapp-`)
5. Go to **OAuth & Permissions** → Add Bot Token Scopes:
   - `chat:write`
   - `app_mentions:read`
   - `channels:history`
   - `im:history`
   - `channels:read`
6. Go to **Event Subscriptions** → Enable Events → Subscribe to bot events:
   - `app_mention`
   - `message.im`
7. Go to **App Home** → Enable "Messages Tab" under "Show Tabs"
8. Install the app to the workspace
9. Share with you:
   - **Bot User OAuth Token** (starts with `xoxb-`)
   - **App-Level Token** (starts with `xapp-`)
10. Invite bot to the target channel: `/invite @Router Config Assistant`

**Option B — Request "App Manager" Permission:**

Ask your workspace admin to grant you the **"App Manager"** role:
- Admin goes to: Workspace Settings → Permissions → Org-level App Management
- Or: https://YOUR-WORKSPACE.slack.com/admin/settings → Permissions → App Management
- Enable: "Allow specific members to install apps"
- Add your account

Once granted, follow the creation steps yourself.

**Option C — Use a Slack Workspace You Control for Development:**

- Create a free Slack workspace at https://slack.com/get-started#/create for development
- Build and test there
- When ready, ask your company admin to install in the production workspace

> [!TIP]
> **Recommendation**: Use Option C for development. Request Option A for production deployment. This way you don't block development while waiting for admin approval.

**Deliverable**: Slack App-Level Token (`xapp-`) and Bot Token (`xoxb-`) secured in `.env`

---

### Task 2.2: Slack Bot Implementation

**File**: `src/slack_bot.py`

**Checklist:**

- [ ] Initialize Slack Bolt app with Socket Mode
- [ ] Implement `@mention` handler for channel messages
- [ ] Implement DM handler
- [ ] Extract clean question text (remove bot mention, Slack formatting)
- [ ] Implement thread context retrieval:
  - [ ] Detect if message is in a thread (`thread_ts` present)
  - [ ] Fetch up to last 3 Q&A pairs from thread via `conversations.replies`
  - [ ] Format as conversation context for the RAG engine
- [ ] Send immediate acknowledgment: "🔍 Looking through the CLI reference docs..."
- [ ] Submit query to async worker pool
- [ ] Post response in same thread (or start new thread)
- [ ] Implement per-user rate limiting (1 query per 10 seconds)
- [ ] Handle errors with user-friendly messages
- [ ] Add structured logging

**Code Structure:**

```python
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from rag_engine import RAGEngine
from config import Config

logger = logging.getLogger(__name__)

# Initialize
app = App(token=Config.SLACK_BOT_TOKEN)
rag_engine = RAGEngine(
    chromadb_path=Config.CHROMADB_PATH,
    model_name=Config.MODEL_NAME,
    embedding_model=Config.EMBEDDING_MODEL
)
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="rag-worker")

# Simple per-user rate limit tracking
_last_query_time: dict[str, float] = {}
RATE_LIMIT_SECONDS = 10

@app.event("app_mention")
def handle_mention(event, say, client):
    """Handle @mentions of the bot in channels."""
    user_id = event["user"]
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])
    
    # Rate limit check
    if _is_rate_limited(user_id):
        say(text="⏳ Please wait a moment before asking another question.", 
            thread_ts=thread_ts)
        return
    
    # Extract question
    question = _extract_question(event["text"], app.client.auth_test()["user_id"])
    if not question.strip():
        say(text="Please include a question after mentioning me!", thread_ts=thread_ts)
        return
    
    # Acknowledge immediately
    say(text="🔍 Looking through the CLI reference docs...", thread_ts=thread_ts)
    
    # Get thread context if in a thread
    thread_context = _get_thread_context(client, channel, thread_ts)
    
    # Submit to worker pool
    executor.submit(_process_query, question, channel, thread_ts, thread_context, client)

@app.event("message")  
def handle_dm(event, say, client):
    """Handle direct messages to the bot."""
    if event.get("channel_type") != "im":
        return
    if event.get("subtype"):  # Ignore message edits, deletes, etc.
        return
    # Similar logic to handle_mention but without stripping bot mention

def _extract_question(text: str, bot_user_id: str) -> str:
    """Remove bot mention and clean up question text."""
    import re
    text = re.sub(f'<@{bot_user_id}>', '', text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove other Slack formatting
    return text.strip()

def _get_thread_context(client, channel: str, thread_ts: str) -> list[dict]:
    """Fetch last 3 Q&A pairs from thread."""
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts, limit=10)
        messages = result.get("messages", [])
        context = []
        for msg in messages[:-1]:  # Exclude current message
            role = "assistant" if msg.get("bot_id") else "user"
            context.append({"role": role, "content": msg.get("text", "")})
        return context[-6:]  # Last 3 pairs (6 messages)
    except Exception as e:
        logger.warning(f"Failed to get thread context: {e}")
        return []

def _process_query(question, channel, thread_ts, thread_context, client):
    """Process query in worker thread and post response."""
    try:
        answer = rag_engine.answer_query(question, thread_context)
        client.chat_postMessage(channel=channel, text=answer, thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        client.chat_postMessage(
            channel=channel,
            text="❌ Sorry, I encountered an error processing your question. Please try again.",
            thread_ts=thread_ts
        )

def _is_rate_limited(user_id: str) -> bool:
    """Check if user has queried too recently."""
    import time
    now = time.time()
    last = _last_query_time.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return True
    _last_query_time[user_id] = now
    return False

if __name__ == "__main__":
    Config.validate()
    handler = SocketModeHandler(app, Config.SLACK_APP_TOKEN)
    logger.info("⚡ Router Config Assistant starting...")
    handler.start()
```

> [!NOTE]
> **Socket Mode vs HTTP**: We use Socket Mode because it doesn't require a public URL or webhook endpoint. The bot connects outbound to Slack's servers via WebSocket. This is ideal for an on-premise VM behind a firewall.

**Deliverable**: Slack bot receiving questions, dispatching to workers, and posting responses in threads

---

## Phase 3: Chainlit Web UI (Week 2-3)

### Task 3.1: Chainlit Integration

[Chainlit](https://github.com/Chainlit/chainlit) is a Python framework purpose-built for LLM chat applications. It provides a ChatGPT-style interface with streaming responses, conversation history, source citations, and dark mode — all out of the box.

**File**: `src/web_ui.py`

```python
import chainlit as cl
from rag_engine import RAGEngine
from config import Config

# Initialize shared RAG engine (same as Slack bot uses)
rag_engine = RAGEngine(
    chromadb_path=Config.CHROMADB_PATH,
    model_name=Config.MODEL_NAME,
    embedding_model=Config.EMBEDDING_MODEL
)

@cl.on_chat_start
async def start():
    """Initialize session with empty history."""
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 **Router Config Assistant**\n\n"
                "Ask me anything about router CLI commands and configurations. "
                "I'll search the CLI reference documentation and provide cited answers.\n\n"
                "_Examples:_\n"
                "- What is the syntax for `show bgp summary`?\n"
                "- How do I configure ISIS authentication?\n"
                "- Show me MPLS-related commands"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Process user message through RAG pipeline."""
    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Get thread history for context
    thread_context = cl.user_session.get("history", [])
    
    # Run RAG query (blocking call wrapped for async)
    answer = await cl.make_async(rag_engine.answer_query)(
        message.content, thread_context
    )
    
    # Update the message with the answer
    msg.content = answer
    await msg.update()
    
    # Update conversation history (keep last 3 Q&A pairs)
    thread_context.append({"role": "user", "content": message.content})
    thread_context.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", thread_context[-6:])
```

**File**: `.chainlit/config.toml`

```toml
[project]
name = "Router Config Assistant"
enable_telemetry = false

[features]
prompt_playground = false       # Not needed for RAG
multi_modal.enabled = false     # No image upload needed

[UI]
name = "Router Config Assistant"
description = "Ask questions about router CLI commands and configurations"
default_theme = "dark"
# Custom CSS can be added via public/stylesheet.css
```

**File**: `chainlit.md` (welcome screen content)

```markdown
# Router Config Assistant 🔧

Welcome! I can help you find information from the router CLI reference documentation.

## What I Can Do
- Look up **command syntax** and parameters
- Find **configuration examples**
- Explain command **options and defaults**
- Search across all CLI reference documents

## Tips for Best Results
- Use specific command names (e.g., "show bgp summary" instead of "show me stuff")
- Ask follow-up questions — I remember our conversation context
- Every answer includes source citations so you can verify

## Limitations
- I only know about commands documented in the CLI reference files
- I cannot execute commands or access live routers
- I cannot make configuration changes
```

> [!NOTE]
> Include `chainlit.md` in the Docker image (copy it in the Dockerfile) so the welcome screen shows in containerized deployments.

**Checklist:**

- [ ] Create `src/web_ui.py` with Chainlit handlers
- [ ] Create `.chainlit/config.toml` with project settings
- [ ] Create `chainlit.md` for welcome screen
- [ ] Test locally: `chainlit run src/web_ui.py -w` (watch mode for development)
- [ ] Verify streaming responses work
- [ ] Verify conversation context carries across messages
- [ ] Verify source citations display correctly
- [ ] Test with the same 20 queries used for Slack bot validation
- [ ] Verify the web UI is accessible at `http://<VM_IP>:8080` from other machines

> [!TIP]
> **Chainlit features you get for free:** Dark/light theme toggle, message copy, conversation history within a session, markdown rendering with code blocks, and a responsive mobile-friendly layout. No custom CSS needed for the basics.

**Deliverable**: ChatGPT-style web UI at `http://<VM_IP>:8080` using the same RAG engine as the Slack bot

---

## Phase 4: Configuration & Dockerization (Week 2-3)

### Task 4.1: Configuration Management

**File**: `src/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Slack (Socket Mode)
    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')       # xoxb-...
    SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')        # xapp-...
    
    # Ollama
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    MODEL_NAME = os.getenv('MODEL_NAME', 'mistral:7b')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
    
    # Paths
    CHROMADB_PATH = os.getenv('CHROMADB_PATH', './data/chromadb')
    ADOC_FILES_PATH = os.getenv('ADOC_FILES_PATH', './data/adoc_files')
    
    # RAG parameters
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 5))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.4))
    MAX_THREAD_CONTEXT = int(os.getenv('MAX_THREAD_CONTEXT', 3))  # Q&A pairs
    
    # Worker pool
    WORKER_THREADS = int(os.getenv('WORKER_THREADS', 3))
    QUERY_TIMEOUT = int(os.getenv('QUERY_TIMEOUT', 45))  # seconds
    RATE_LIMIT_SECONDS = int(os.getenv('RATE_LIMIT_SECONDS', 10))
    
    @classmethod
    def validate(cls):
        """Validate required configuration at startup."""
        required = {
            'SLACK_BOT_TOKEN': cls.SLACK_BOT_TOKEN,
            'SLACK_APP_TOKEN': cls.SLACK_APP_TOKEN,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
```

**.env.example:**

```bash
# ──── Slack (Socket Mode) ────
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-level-token-here

# ──── Ollama ────
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=mistral:7b
EMBEDDING_MODEL=nomic-embed-text

# ──── Paths ────
CHROMADB_PATH=./data/chromadb
ADOC_FILES_PATH=./data/adoc_files

# ──── RAG Tuning ────
CHUNK_SIZE=1000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.4
MAX_THREAD_CONTEXT=3

# ──── Performance ────
WORKER_THREADS=3
QUERY_TIMEOUT=45
RATE_LIMIT_SECONDS=10
```

**Deliverable**: Clean configuration module with validation

---

### Task 4.2: Docker Setup

**File**: `Dockerfile`

```dockerfile
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY chainlit.md ./chainlit.md

RUN chown -R appuser:appuser /app
USER appuser

# Health check — verify Python can import all modules
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.config import Config; print('ok')" || exit 1

CMD ["python", "src/slack_bot.py"]
```

**File**: `.dockerignore`

```
.env
.git
__pycache__
*.pyc
venv/
data/
tests/
docs/
backups/
.pytest_cache/
*.md
!chainlit.md
```

**File**: `docker-compose.yml`

```yaml
services:
  slack-bot:
    build: .
    container_name: router-config-slack-bot
    command: python src/slack_bot.py
    env_file: .env
    environment:
      OLLAMA_HOST: http://host.docker.internal:11434
      CHROMADB_PATH: /app/data/chromadb
      ADOC_FILES_PATH: /app/data/adoc_files
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./data/chromadb:/app/data/chromadb:ro
      - ./data/adoc_files:/app/data/adoc_files:ro
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
    deploy:
      resources:
        limits:
          memory: 4G

  web-ui:
    build: .
    container_name: router-config-web-ui
    command: chainlit run src/web_ui.py --host 0.0.0.0 --port 8000
    ports:
      - "8080:8000"
    env_file: .env
    environment:
      OLLAMA_HOST: http://host.docker.internal:11434
      CHROMADB_PATH: /app/data/chromadb
      ADOC_FILES_PATH: /app/data/adoc_files
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./data/chromadb:/app/data/chromadb:ro
      - ./data/adoc_files:/app/data/adoc_files:ro
      - ./.chainlit:/app/.chainlit
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
    deploy:
      resources:
        limits:
          memory: 4G
```

> [!NOTE]
> **Two-service architecture.** The Slack bot and Chainlit web UI run as separate containers, but share the same RAG engine code and ChromaDB volume. Ollama runs directly on the host (not in Docker) for best CPU performance. Both containers connect to host Ollama via `host.docker.internal`.

**Checklist:**

- [ ] Create `Dockerfile`
- [ ] Create `.dockerignore`
- [ ] Create `docker-compose.yml`
- [ ] Build image: `docker compose build`
- [ ] Test locally: `docker compose up`
- [ ] Verify bot connects to Slack
- [ ] Verify connection to host Ollama from inside container
- [ ] Test a query end-to-end through Docker

**Deliverable**: Dockerized application with two-service deployment

---

### Task 4.3: Operations Makefile

**File**: `Makefile`

```makefile
.PHONY: help setup build start stop restart logs rebuild-index test backup

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial setup — create dirs, copy env template
	@mkdir -p data/adoc_files data/chromadb backups
	@test -f .env || cp .env.example .env
	@echo "✅ Setup complete. Edit .env with your tokens, then run 'make rebuild-index'"

build: ## Build Docker image
	docker compose build

start: ## Start the assistant
	docker compose up -d
	@echo "✅ Assistant started. Check logs with 'make logs'"

stop: ## Stop the assistant
	docker compose down

restart: ## Restart all services
	docker compose restart

restart-slack: ## Restart Slack bot only
	docker compose restart slack-bot

restart-webui: ## Restart web UI only
	docker compose restart web-ui

logs: ## Tail all logs
	docker compose logs -f

logs-slack: ## Tail Slack bot logs only
	docker compose logs -f slack-bot

logs-webui: ## Tail web UI logs only
	docker compose logs -f web-ui

rebuild-index: ## Rebuild ChromaDB index from adoc files
	@echo "⏳ Rebuilding index from data/adoc_files/..."
	python3 scripts/rebuild_index.py
	@echo "✅ Index rebuilt"

rebuild-index-dryrun: ## Preview chunking without storing
	python3 scripts/rebuild_index.py --dry-run

test: ## Run tests
	python -m pytest tests/ -v

backup: ## Backup ChromaDB index
	@mkdir -p backups
	tar -czf backups/chromadb_$$(date +%Y%m%d_%H%M%S).tar.gz data/chromadb/
	@echo "✅ Backup saved to backups/"

ollama-status: ## Check Ollama status and loaded models
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool
	@echo ""
	@curl -s http://localhost:11434/api/ps | python3 -m json.tool
```

**Deliverable**: Developer-friendly command interface

---

## Phase 5: Testing & Tuning (Week 3)

### Task 5.1: Unit Tests

**Checklist:**

- [ ] Set up pytest with `conftest.py` for shared fixtures
- [ ] Test AsciiDoc parser:
  - [ ] Heading detection and hierarchy
  - [ ] Code block preservation (never split mid-block)
  - [ ] Table preservation
  - [ ] Protocol detection
- [ ] Test document processor:
  - [ ] File discovery in directory
  - [ ] Chunk generation and metadata
  - [ ] Embedding generation (mock Ollama for unit tests)
  - [ ] ChromaDB storage and retrieval
- [ ] Test RAG engine:
  - [ ] Query preprocessing (Slack formatting removal)
  - [ ] Semantic search with mock data
  - [ ] Prompt construction (verify structure)
  - [ ] Response formatting for Slack
  - [ ] Edge cases (no results, timeout)
- [ ] Test Slack bot:
  - [ ] Rate limiting logic
  - [ ] Question extraction from mentions
  - [ ] Thread context retrieval (mock Slack API)
- [ ] Achieve >80% code coverage

**File**: `tests/conftest.py`

```python
import pytest
import tempfile
import chromadb

@pytest.fixture
def sample_adoc_content():
    return """= show bgp summary
:description: Display BGP summary information

== Syntax
  show bgp summary [vrf <vrf-name>]

== Parameters
|===
| Parameter | Description
| vrf       | VRF name to filter (optional)
|===

== Examples
----
router# show bgp summary
BGP router identifier 10.0.0.1, local AS number 65000
Neighbor        AS    MsgRcvd   MsgSent   Up/Down    State
10.0.0.2        65001 1234      5678      01:23:45   Established
----

== Related Commands
- show bgp neighbors
- show bgp routes
"""

@pytest.fixture
def tmp_chromadb():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        yield client, tmpdir
```

**Deliverable**: Comprehensive test suite with >80% coverage

---

### Task 5.2: Integration Testing

**Checklist:**

- [ ] Test full pipeline: question → RAG → answer (without Slack)
- [ ] Test with 20+ real CLI reference questions:
  - [ ] "What is the syntax for show bgp summary?"
  - [ ] "How do I configure ISIS authentication with MD5?"
  - [ ] "Show me all MPLS-related commands"
  - [ ] "What parameters does the show interface command accept?"
  - [ ] "Give me an example of configuring BGP neighbors"
- [ ] Test hallucination resistance:
  - [ ] "How do I configure Kubernetes on this router?" (should say "not found")
  - [ ] "What is the command for WiFi setup?" (should say "not found")
  - [ ] Ask about a real command but with fake parameters
- [ ] Test thread context:
  - [ ] Ask "show bgp summary syntax" → follow up with "what about the vrf option?"
  - [ ] Verify second answer uses context from first
- [ ] Measure response times for 20 queries, compute p50/p95/p99
- [ ] Test concurrent queries: submit 5 at once, verify all complete correctly
- [ ] Validate citation accuracy: every answer should cite the correct source file

**Deliverable**: Validated system with measured performance baselines

---

### Task 5.3: Prompt Engineering & Parameter Tuning

**Checklist:**

- [ ] Test chunk sizes: 500, 800, 1000, 1500 characters
  - [ ] Measure retrieval quality at each size
  - [ ] Find optimal size for CLI reference structure
- [ ] Test top-K values: 3, 5, 7
  - [ ] Too few = missing info; too many = context pollution
- [ ] Test similarity thresholds: 0.3, 0.4, 0.5, 0.6
  - [ ] Too low = irrelevant results; too high = missing relevant results
- [ ] Refine system prompt iteratively:
  - [ ] Test with trick questions
  - [ ] Ensure code blocks render correctly in Slack
  - [ ] Verify citation format consistency
- [ ] Compare `mistral:7b` vs `gemma2:2b` on your benchmark queries:
  - [ ] Quality score (manual 1-5)
  - [ ] Response time
  - [ ] Choose the winner
- [ ] Test `temperature` settings: 0.0, 0.1, 0.2
  - [ ] Lower = more deterministic (good for reference docs)
- [ ] Document final optimal parameters in `.env.example`

**Deliverable**: Tuned parameters and documented benchmark results

---

## Phase 6: Deployment & Documentation (Week 4)

### Task 6.1: Deployment Scripts

Two scripts handle deployment: a **base setup** script for fresh servers, and a **deploy** script for the application itself.

#### Script 1: `scripts/base_setup.sh` — Server Prerequisites

Run once on a fresh Debian/Ubuntu server. Installs Docker, Ollama, and Python.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════
# base_setup.sh — Install prerequisites on a fresh server
# Run as: sudo bash scripts/base_setup.sh
# ═══════════════════════════════════════════════════════

echo "══════════════════════════════════════════════════"
echo "  Router Config Assistant — Base Server Setup"
echo "══════════════════════════════════════════════════"

# ── Check root ──
if [[ $EUID -ne 0 ]]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

# ── 1. System packages ──
echo ""
echo "📦 [1/4] Installing system packages..."
apt-get update -qq
apt-get install -y -qq curl git python3 python3-pip python3-venv \
    ca-certificates gnupg lsb-release

# ── 2. Docker ──
echo ""
echo "🐳 [2/4] Installing Docker..."
if command -v docker &>/dev/null; then
    echo "  ✅ Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker && systemctl start docker
    echo "  ✅ Docker installed: $(docker --version)"
fi

# Add sudo user to docker group
if [ -n "${SUDO_USER:-}" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "  ℹ️  Added $SUDO_USER to docker group (re-login required)"
fi

# Verify Docker Compose plugin
if ! docker compose version &>/dev/null; then
    apt-get install -y -qq docker-compose-plugin
fi
echo "  ✅ Docker Compose: $(docker compose version --short)"

# ── 3. Ollama ──
echo ""
echo "🤖 [3/4] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  ✅ Ollama already installed"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi

systemctl enable ollama && systemctl start ollama
sleep 3

echo "  📥 Pulling mistral:7b (this may take several minutes)..."
ollama pull mistral:7b
echo "  📥 Pulling nomic-embed-text (default EMBEDDING_MODEL)..."
ollama pull nomic-embed-text

# ── 4. Configure Ollama for parallel requests ──
echo ""
echo "⚙️  [4/4] Configuring Ollama for parallel inference..."
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=3"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
EOF
systemctl daemon-reload && systemctl restart ollama

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Base setup complete!"
echo "  Next: Run 'bash scripts/deploy.sh' (as non-root)"
echo "══════════════════════════════════════════════════"
```

---

#### Script 2: `scripts/deploy.sh` — Application Deployment

Run after `base_setup.sh`. Validates prerequisites, configures `.env` (from file or interactive prompts), copies adoc files, builds index, and starts services.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════
# deploy.sh — Deploy the Router Config Assistant
# Usage: bash scripts/deploy.sh [adoc_files_path]
# ═══════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "══════════════════════════════════════════════════"
echo "  Router Config Assistant — Deployment"
echo "══════════════════════════════════════════════════"

# ── 1. Validate prerequisites ──
echo ""
echo "🔍 [1/6] Checking prerequisites..."
ERRORS=()
if [ -f .env ]; then
    set -a
    . .env
    set +a
fi
EMBEDDING_MODEL=${EMBEDDING_MODEL:-nomic-embed-text}
command -v docker &>/dev/null       || ERRORS+=("Docker not installed")
docker compose version &>/dev/null  || ERRORS+=("Docker Compose not found")
command -v ollama &>/dev/null       || ERRORS+=("Ollama not installed")
curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 || ERRORS+=("Ollama not running")
ollama list 2>/dev/null | grep -q "mistral"       || ERRORS+=("mistral:7b not pulled")
ollama list 2>/dev/null | grep -q "${EMBEDDING_MODEL}"    || ERRORS+=("${EMBEDDING_MODEL} not pulled")

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "  ❌ Failed:"
    printf '     • %s\n' "${ERRORS[@]}"
    echo "  Run 'sudo bash scripts/base_setup.sh' first."
    exit 1
fi
echo "  ✅ All prerequisites met"

# ── 2. Create directories ──
echo ""
echo "📁 [2/6] Setting up directories..."
mkdir -p data/adoc_files data/chromadb backups .chainlit

# ── 3. Configure .env ──
echo ""
echo "🔑 [3/6] Configuring environment..."
if [ -f .env ]; then
    echo "  ✅ Using existing .env file"
else
    cp .env.example .env
    echo "  No .env found — creating from template."
    echo "  (Leave blank to skip Slack and use only the web UI)"
    echo ""
    read -rp "  Slack Bot Token (xoxb-...): " SLACK_BOT_TOKEN
    read -rp "  Slack App Token (xapp-...): " SLACK_APP_TOKEN
    sed -i "s|SLACK_BOT_TOKEN=.*|SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN}|" .env
    sed -i "s|SLACK_APP_TOKEN=.*|SLACK_APP_TOKEN=${SLACK_APP_TOKEN}|" .env
    echo "  ✅ .env configured"
fi

# ── 4. Copy adoc files ──
echo ""
echo "📄 [4/6] Setting up documentation files..."
ADOC_PATH="${1:-}"
if [ -n "$ADOC_PATH" ] && [ -d "$ADOC_PATH" ]; then
    cp "$ADOC_PATH"/*.adoc data/adoc_files/ 2>/dev/null || true
fi
FILE_COUNT=$(find data/adoc_files/ -name "*.adoc" 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -gt 0 ]; then
    echo "  ✅ Found $FILE_COUNT .adoc files"
else
    echo "  ⚠️  No .adoc files in data/adoc_files/"
    read -rp "  Path to adoc files (or Enter to skip): " MANUAL_PATH
    if [ -n "$MANUAL_PATH" ] && [ -d "$MANUAL_PATH" ]; then
        cp "$MANUAL_PATH"/*.adoc data/adoc_files/ 2>/dev/null || true
        FILE_COUNT=$(find data/adoc_files/ -name "*.adoc" | wc -l)
        echo "  ✅ Copied $FILE_COUNT files"
    else
        echo "  ⏭️  Skipped — run 'make rebuild-index' after copying files"
    fi
fi

# ── 5. Build ──
echo ""
echo "🏗️  [5/6] Building Docker images..."
docker compose build --quiet
echo "  ✅ Images built"

# Build index if files exist
FILE_COUNT=$(find data/adoc_files/ -name "*.adoc" 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -gt 0 ]; then
    echo ""
    echo "📊 Building search index from $FILE_COUNT files..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    venv/bin/pip install -r requirements.txt
    venv/bin/python scripts/rebuild_index.py
    echo "  ✅ Index built"
fi

# ── 6. Start ──
echo ""
echo "🚀 [6/6] Starting services..."
docker compose up -d

VM_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  📎 Web UI:    http://${VM_IP}:8080"
echo "  📎 Slack Bot:  @Router Config Assistant"
echo ""
echo "  🔧 Commands:  make logs | make stop | make rebuild-index"
echo "══════════════════════════════════════════════════"
```

**Usage:**

```bash
# Fresh server — full setup:
sudo bash scripts/base_setup.sh
bash scripts/deploy.sh /path/to/Command_Line_Interface

# Prerequisites already installed:
bash scripts/deploy.sh /path/to/Command_Line_Interface

# Adoc files already in data/adoc_files/:
bash scripts/deploy.sh
```

**Checklist:**

- [ ] Create `scripts/base_setup.sh`
- [ ] Create `scripts/deploy.sh`
- [ ] Test `base_setup.sh` on a fresh Debian 12 VM
- [ ] Test `deploy.sh` with adoc path argument
- [ ] Test `deploy.sh` with interactive `.env` prompts (no existing `.env`)
- [ ] Test `deploy.sh` with existing `.env` (skips prompts)
- [ ] Verify both services start and respond correctly
- [ ] Test `http://<VM_IP>:8080` from another machine on the network
- [ ] Set up auto-restart on boot:
  ```bash
  @reboot cd /opt/router-config-assistant && docker compose up -d
  ```
- [ ] Create weekly backup cron:
  ```bash
  0 2 * * 0 cd /opt/router-config-assistant && make backup
  ```

**Deliverable**: One-command deployment on a fresh server

---

### Task 6.2: Documentation

**Checklist:**

- [ ] Create `README.md` with:
  - [ ] Project overview
  - [ ] Quick start guide
  - [ ] Architecture diagram
  - [ ] Available `make` commands
- [ ] Create `docs/user_guide.md`:
  - [ ] How to access the bot (mention in channel or DM)
  - [ ] Example queries with expected response format
  - [ ] Thread usage for follow-ups
  - [ ] Tips for best results
  - [ ] Known limitations
- [ ] Create `docs/admin_guide.md`:
  - [ ] Architecture overview
  - [ ] How to update adoc files and rebuild index
  - [ ] How to restart / troubleshoot
  - [ ] How to change model or tuning parameters
  - [ ] Backup and recovery
  - [ ] Log analysis
- [ ] Create `docs/update_procedure.md`:
  - [ ] Step-by-step for updating docs (backup → copy → rebuild → verify)
  - [ ] Rollback procedure

**Deliverable**: Complete documentation suite

---

## Phase 7: Monitoring & Future Enhancements (Post-Launch)

### Task 7.1: Structured Logging

- [ ] Implement JSON-structured logging across all modules
- [ ] Log every query with: `correlation_id`, `user_id`, `question`, `retrieval_time`, `inference_time`, `total_time`, `num_chunks_retrieved`
- [ ] Log errors with full context
- [ ] Create a simple log analysis script: `scripts/analyze_logs.py`
  - [ ] Average response time
  - [ ] Error rate
  - [ ] Most common queries
  - [ ] Busiest hours

### Task 7.2: Future Enhancements (Backlog)

| Priority | Enhancement | Value |
|---|---|---|
| High | Query result caching (TTL-based) | Faster repeat queries |
| High | Feedback mechanism (👍/👎 reactions) | Quality tracking |
| Medium | Web UI authentication (SSO/LDAP) | Broader internal access |
| Medium | Prometheus + Grafana dashboards | Operational visibility |
| Medium | Query auto-suggest / "did you mean" | Better UX |
| Low | Multi-channel Slack support | Broader access |
| Low | Export config snippets to file | Power user feature |
| Low | Open WebUI migration | More feature-rich web interface |

---

## Project Structure

```
router-config-assistant/
├── README.md
├── Makefile                       # Developer commands
├── .env.example                   # Configuration template
├── .env                           # Actual config (gitignored)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml             # Slack bot + Web UI services
├── requirements.txt
├── chainlit.md                    # Chainlit welcome screen
│
├── .chainlit/
│   └── config.toml                # Chainlit UI configuration
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration loading & validation
│   ├── document_processor.py      # AsciiDoc parsing & indexing
│   ├── rag_engine.py              # RAG query pipeline
│   ├── slack_bot.py               # Slack bot + async worker pool
│   └── web_ui.py                  # Chainlit web UI handler
│
├── scripts/
│   ├── base_setup.sh              # Server prerequisites (Docker, Ollama)
│   ├── deploy.sh                  # Application deployment
│   ├── rebuild_index.py           # Rebuild ChromaDB index
│   ├── benchmark.py               # Performance benchmarking
│   ├── test_ollama.py             # Test Ollama connection
│   └── analyze_logs.py            # Log analysis
│
├── data/
│   ├── adoc_files/                # Mount: CLI reference .adoc files
│   └── chromadb/                  # Persistent vector store
│
├── tests/
│   ├── conftest.py                # Shared test fixtures
│   ├── test_document_processor.py
│   ├── test_rag_engine.py
│   └── test_slack_bot.py
│
├── docs/
│   ├── user_guide.md
│   ├── admin_guide.md
│   └── update_procedure.md
│
└── backups/                       # ChromaDB backups
```

---

## Critical Implementation Notes

### 1. Response Time Budget (32-core CPU, 64GB RAM)

| Step | Estimated Time |
|---|---|
| Query preprocessing | <50ms |
| Embedding generation (`EMBEDDING_MODEL`, default `nomic-embed-text`) | 200-500ms |
| ChromaDB similarity search | 50-100ms |
| LLM inference (`mistral:7b` on 32 cores) | **5-10s** |
| Response formatting | <50ms |
| **Total** | **6-11 seconds** |

> [!TIP]
> Your 32-core CPU is well-suited for `mistral:7b`. If you do find it too slow, switch to `gemma2:2b` for ~3-6s total — just change `MODEL_NAME` in `.env`.

### 2. Hallucination Prevention (4-Layer Strategy)

```
Layer 1: System Prompt          → "ONLY use provided documentation"
Layer 2: Retrieval Threshold    → Reject if best similarity < 0.4
Layer 3: Citation Requirement   → Prompt demands citations, post-processing validates
Layer 4: Temperature Control    → temperature=0.1 for deterministic, factual responses
```

### 3. Thread Context Budget

With `mistral:7b`'s 32K context window:
- System prompt: ~300 tokens
- Retrieved chunks (5 × ~300 tokens): ~1500 tokens
- Thread context (3 Q&A pairs): ~600 tokens
- User question: ~50 tokens
- **Total input: ~2500 tokens** → plenty of headroom

### 4. Error Handling

```python
ERROR_MESSAGES = {
    'no_results': "🤔 I couldn't find information about that in the CLI reference files. "
                  "Try asking about a specific command name.",
    'timeout':    "⏳ I'm taking longer than expected. Please try again in a moment.",
    'service_down': "⚠️ I'm having trouble connecting to my knowledge base. "
                    "Please try again in a few minutes.",
    'rate_limited': "⏳ Please wait a moment before asking another question.",
}
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Slow responses (>15s) | Low (32 cores) | Medium | Fall back to `gemma2:2b`; cache frequent queries |
| Hallucinations | Medium | High | 4-layer prevention; regular testing |
| Ollama crashes | Low | High | `systemctl` auto-restart; health monitoring |
| Stale documentation | Low | Medium | Clear update procedure; version tracking |
| Slack token rotation | Low | Low | Document re-authentication process |

---

## Success Metrics

### Week 1-2
- [ ] RAG pipeline working with test data
- [ ] >80% retrieval accuracy on 20 sample queries
- [ ] Response time <12s on 32-core CPU

### Week 3
- [ ] Slack bot functional in test workspace
- [ ] Docker deployment working
- [ ] Unit + integration tests passing

### Week 4
- [ ] Production deployment on target VM
- [ ] Documentation published
- [ ] 10+ successful queries from real users

### Ongoing
- [ ] <3% error rate
- [ ] Average response time <12s
- [ ] Zero hallucinations in monitored queries
- [ ] User satisfaction feedback positive

---

## Quick Reference: Key Differences from Original Plan

| Aspect | Original Plan | Enhanced Plan | Reason |
|---|---|---|---|
| **Architecture** | Slack → RabbitMQ → Workers | Slack + Web UI → ThreadPoolExecutor | 5 users, no need for MQ overhead |
| **LLM** | `gemma2:2b` (primary) | `mistral:7b` (primary) | 32-core/64GB VM can handle it, much better quality |
| **Slack Auth** | HTTP Events API | Socket Mode | No public URL needed on-premise |
| **Tokens** | `SLACK_SIGNING_SECRET` | `SLACK_APP_TOKEN` | Socket Mode uses app-level token |
| **Chunking** | Generic 800-char splits | Section-aware (heading-based) | CLI reference docs have clear structure |
| **Dependencies** | LangChain + pika + asciidoc | Direct APIs + Chainlit | Simpler, fewer moving parts |
| **Containers** | 3 services (MQ + bot + workers) | 2 services (slack-bot + web-ui) | Simpler ops |
| **Web Interface** | None | Chainlit (ChatGPT-style) | Browser access in addition to Slack |
| **Deployment** | Manual steps | Automated scripts (base_setup.sh + deploy.sh) | One-command setup on fresh server |
| **Thread context** | 5 Q&A pairs | 3 Q&A pairs | Fit in context window safely |
| **Framework** | LangChain | Raw Ollama + ChromaDB APIs | Transparency, simplicity |
