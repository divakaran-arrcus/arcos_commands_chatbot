"""Slack Bot for Router Config RAG Assistant.

This module provides a Slack bot interface using Socket Mode.
No public URL or firewall changes required.
"""
import sys
import time
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Add project root to Python path so 'src' module can be found
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.rag_engine import RAGEngine
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables (initialized in main())
app = None
rag_engine = None
executor = None
_last_query_time: dict[str, float] = {}
RATE_LIMIT_SECONDS = Config.RATE_LIMIT_SECONDS


def _is_rate_limited(user_id: str) -> bool:
    """Check if user has queried too recently."""
    now = time.time()
    last = _last_query_time.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return True
    _last_query_time[user_id] = now
    return False


def _extract_question(text: str, bot_user_id: str) -> str:
    """Remove bot mention and clean up question text."""
    # Remove bot mention <@USER_ID>
    text = re.sub(f'<@{bot_user_id}>', '', text)
    # Remove other Slack formatting (links, user mentions, etc.)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def _get_thread_context(client, channel: str, thread_ts: str) -> list[dict]:
    """Fetch last N Q&A pairs from thread for context."""
    try:
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=10  # Get last 10 messages
        )
        messages = result.get("messages", [])
        context = []
        
        # Get bot user ID to identify bot messages
        bot_user_id = client.auth_test()["user_id"]
        
        for msg in messages[:-1]:  # Exclude current message
            # Bot messages have bot_id, user messages have user
            if msg.get("bot_id") or msg.get("user") == bot_user_id:
                role = "assistant"
                content = msg.get("text", "")
            else:
                role = "user"
                content = msg.get("text", "")
            
            # Clean the content
            content = _extract_question(content, bot_user_id) if role == "user" else content
            if content:
                context.append({"role": role, "content": content})
        
        # Return last N Q&A pairs (2 messages per pair)
        max_messages = Config.MAX_THREAD_CONTEXT * 2
        return context[-max_messages:]
        
    except Exception as e:
        logger.warning(f"Failed to get thread context: {e}")
        return []


def _process_query(question: str, channel: str, thread_ts: str, 
                   thread_context: list[dict], client):
    """Process query in worker thread and post response."""
    try:
        logger.info(f"Processing query: {question[:100]}...")
        answer = rag_engine.answer_query(question, thread_context)
        
        # Post response in the same thread
        client.chat_postMessage(
            channel=channel,
            text=answer,
            thread_ts=thread_ts
        )
        logger.info(f"Response posted successfully for query in channel {channel}")
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        client.chat_postMessage(
            channel=channel,
            text="❌ Sorry, I encountered an error processing your question. Please try again.",
            thread_ts=thread_ts
        )


def handle_mention(event, say, client):
    """Handle @mentions of the bot in channels."""
    user_id = event["user"]
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])
    
    logger.info(f"Received mention from user {user_id} in channel {channel}")
    
    # Rate limit check
    if _is_rate_limited(user_id):
        say(
            text=f"⏳ Please wait {RATE_LIMIT_SECONDS} seconds before asking another question.",
            thread_ts=thread_ts
        )
        return
    
    # Extract question text
    bot_user_id = client.auth_test()["user_id"]
    question = _extract_question(event["text"], bot_user_id)
    
    if not question.strip():
        say(
            text="Please include a question after mentioning me!",
            thread_ts=thread_ts
        )
        return
    
    # Send immediate acknowledgment
    say(
        text="🔍 Looking through the CLI reference docs...",
        thread_ts=thread_ts
    )
    
    # Get thread context if in a thread
    thread_context = _get_thread_context(client, channel, thread_ts)
    
    # Submit to worker pool for async processing
    executor.submit(_process_query, question, channel, thread_ts, thread_context, client)


def handle_dm(event, say, client):
    """Handle direct messages to the bot."""
    # Only handle direct messages
    if event.get("channel_type") != "im":
        return
    
    # Ignore message edits, deletes, bot messages, etc.
    if event.get("subtype") or event.get("bot_id"):
        return
    
    user_id = event["user"]
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])
    
    logger.info(f"Received DM from user {user_id}")
    
    # Rate limit check
    if _is_rate_limited(user_id):
        say(
            text=f"⏳ Please wait {RATE_LIMIT_SECONDS} seconds before asking another question.",
            thread_ts=thread_ts
        )
        return
    
    # For DMs, the entire message is the question
    question = event.get("text", "").strip()
    
    if not question:
        say(
            text="Please include a question!",
            thread_ts=thread_ts
        )
        return
    
    # Send immediate acknowledgment
    say(
        text="🔍 Looking through the CLI reference docs...",
        thread_ts=thread_ts
    )
    
    # Get thread context if in a thread
    thread_context = _get_thread_context(client, channel, thread_ts)
    
    # Submit to worker pool for async processing
    executor.submit(_process_query, question, channel, thread_ts, thread_context, client)


def main():
    """Start the Slack bot."""
    global app, rag_engine, executor
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in your .env file")
        sys.exit(1)
    
    # Check Slack tokens are present
    if not Config.is_slack_enabled():
        logger.error("Slack bot is not enabled. Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env")
        sys.exit(1)
    
    logger.info("⚡ Router Config Assistant starting...")
    logger.info(f"Using model: {Config.MODEL_NAME}")
    logger.info(f"Worker threads: {Config.WORKER_THREADS}")
    
    # Initialize shared RAG engine
    rag_engine = RAGEngine(
        chromadb_path=Config.CHROMADB_PATH,
        model_name=Config.MODEL_NAME,
        embedding_model=Config.EMBEDDING_MODEL
    )
    
    # Initialize Slack Bolt app
    app = App(token=Config.SLACK_BOT_TOKEN)
    
    # Thread pool for concurrent query processing
    executor = ThreadPoolExecutor(
        max_workers=Config.WORKER_THREADS, 
        thread_name_prefix="rag-worker"
    )
    
    # Register event handlers
    app.event("app_mention")(handle_mention)
    app.event("message")(handle_dm)
    
    # Start Socket Mode handler
    handler = SocketModeHandler(app, Config.SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
