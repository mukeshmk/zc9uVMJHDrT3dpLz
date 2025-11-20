"""
Streamlit-based chat interface for the conversational AI agent.

This module provides a web UI for:
- Creating new chat sessions
- Sending messages and receiving responses
- Switching between existing conversations
- Loading conversation history
"""

import asyncio
import logging
import streamlit as st
from uuid import UUID
from datetime import datetime
from typing import Optional, Generator
from contextlib import contextmanager

# Fix for nested event loops in Streamlit
import nest_asyncio
nest_asyncio.apply()

from convai.services.chat import chat_service
from convai.data.schemas import ChatMessage
from convai.utils.logger import setup_logs
from convai.data.database import SessionLocal

# Configure logging for Streamlit app
logger = logging.getLogger("convai")
if not logger.handlers:
    logger = setup_logs(logger)
    logger.info("Starting Streamlit Chat UI")


@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%I:%M %p")


def format_session_label(session_id: UUID, created_at: datetime, messages: list) -> str:
    """Format session label for sidebar."""
    msg_count = len(messages)
    time_str = created_at.strftime("%b %d, %I:%M %p")
    
    # Get first user message as preview if available
    preview = "New Chat"
    for msg in messages:
        if msg.role == "user":
            preview = msg.content[:30] + "..." if len(msg.content) > 30 else msg.content
            break
    
    return f"{preview}\n{time_str} • {msg_count} msgs"


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "awaiting_first_message" not in st.session_state:
        st.session_state.awaiting_first_message = True


def render_sidebar():
    """Render the sidebar with session management."""
    with st.sidebar:
        st.title("💬 Conversations")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            logger.info("User clicked 'New Chat' button - resetting to greeting screen")
            st.session_state.current_session_id = None
            st.session_state.awaiting_first_message = True
            st.rerun()
        
        st.divider()
        
        # List existing sessions
        with get_db_session() as db:
            sessions = chat_service.list_sessions(db)
            
            if sessions:
                st.subheader("Your Chats")                
                for session in sessions:
                    # We need to get messages for the label preview
                    # This might be N+1 query issue but for a simple UI it's fine for now.
                    # TODO: Optimization - Add preview/count to the list_sessions query or better a redis cache.
                    history = chat_service.get_session_history(session.session_id, limit=1, db=db)
                    messages = history.messages
                    
                    label = format_session_label(session.session_id, session.created_at, messages)
                    
                    # Highlight current session
                    button_type = "primary" if session.session_id == st.session_state.current_session_id else "secondary"
                    
                    if st.button(
                        label,
                        key=f"session_{session.session_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        logger.info(f"User switched to session: {session.session_id}")
                        st.session_state.current_session_id = session.session_id
                        st.session_state.awaiting_first_message = False
                        st.rerun()
            else:
                st.info("No conversations yet.\nStart a new chat!")


def render_greeting():
    """Render greeting screen for new users."""
    st.title("🎬 Movie Conversational AI")
    
    st.markdown("""
    ### Welcome! 👋
    
    I'm your AI assistant for all things movies. I can help you with:
    
    - 🎯 **Movie Recommendations** - Find your next favorite film
    - 🔍 **Movie Information** - Get details about specific movies
    - 🎭 **Genre Exploration** - Discover movies by genre
    - ⭐ **Top Rated Movies** - Find the best-rated films
    - 🌦️ **Weather-based Suggestions** - Get movie recommendations based on the weather
    - 💬 **General Questions** - Ask me anything about movies!
    
    ---
    
    **Ready to start?** Type your message below to begin a conversation!
    """)
    
    # Chat input for starting new conversation
    user_input = st.chat_input("Ask me about movies...")
    
    if user_input:
        logger.info(f"User sent message from greeting screen: '{user_input[:50]}...'")
        
        with get_db_session() as db:
            # Create new session
            response = chat_service.create_session(db)
            logger.info(f"Created new session: {response.session_id}")
            st.session_state.current_session_id = response.session_id
            st.session_state.awaiting_first_message = False
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
                st.caption(format_timestamp(datetime.now()))
            
            # Process message and get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Run async function in event loop
                        response = asyncio.run(
                            chat_service.process_message(
                                st.session_state.current_session_id,
                                user_input,
                                db
                            )
                        )
                        st.markdown(response.assistant_response)
                        st.caption(format_timestamp(response.timestamp))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Rerun to update sidebar and continue chat
        st.rerun()


def render_chat_interface():
    """Render the main chat interface."""
    session_id = st.session_state.current_session_id
    
    messages = []
    if session_id:
        with get_db_session() as db:
            # Get conversation history
            history = chat_service.get_session_history(session_id, limit=100, db=db)
            messages = history.messages
    
    # Display chat title
    if session_id and messages:
        st.title("💬 Chat")
    else:
        st.title("🎬 Movie Conversational AI")
    
    # Display conversation history
    if messages:
        for msg in messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                st.caption(format_timestamp(msg.timestamp))
    
    # Chat input
    user_input = st.chat_input("Ask me about movies...")
    
    if user_input:
        logger.info(f"User sent message in session {st.session_state.current_session_id}: '{user_input[:50]}...'")
        
        with get_db_session() as db:
            # Create new session if needed (shouldn't happen here but safe to check)
            if st.session_state.current_session_id is None:
                response = chat_service.create_session(db)
                logger.info(f"Created new session: {response.session_id}")
                st.session_state.current_session_id = response.session_id
                st.session_state.awaiting_first_message = False
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
                st.caption(format_timestamp(datetime.now()))
            
            # Process message and get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Run async function in event loop
                        response = asyncio.run(
                            chat_service.process_message(
                                st.session_state.current_session_id,
                                user_input,
                                db
                            )
                        )
                        st.markdown(response.assistant_response)
                        st.caption(format_timestamp(response.timestamp))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Rerun to update sidebar and clear input
        st.rerun()


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Movie AI Chat",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    if st.session_state.awaiting_first_message:
        render_greeting()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main()
