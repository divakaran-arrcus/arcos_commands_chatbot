"""Chainlit Web UI for Router Config RAG Assistant.

This module provides a ChatGPT-style web interface using Chainlit.
"""
import chainlit as cl
from chainlit import user_session
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
    user_session.set("history", [])
    await cl.Message(
        content="""👋 **Router Config Assistant**

Ask me anything about router CLI commands and configurations. I'll search the CLI reference documentation and provide cited answers.

_Examples:_
- What is the syntax for `show bgp summary`?
- How do I configure ISIS authentication?
- Show me MPLS-related commands"""
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Process user message through RAG pipeline."""
    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    # Get thread history for context
    thread_context = user_session.get("history", [])

    # Run RAG query (blocking call wrapped for async)
    answer = await cl.make_async(rag_engine.answer_query)(
        message.content, thread_context
    )

    # Update the message with the answer
    msg.content = answer
    await msg.update()

    # Update conversation history (keep last 3 Q&A pairs = 6 messages)
    thread_context.append({"role": "user", "content": message.content})
    thread_context.append({"role": "assistant", "content": answer})
    user_session.set("history", thread_context[-6:])


if __name__ == "__main__":
    # Run with: chainlit run src/web_ui.py -w
    import chainlit
    chainlit.run()
