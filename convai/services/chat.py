import logging
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import HTTPException, status
from convai.utils.config import settings
from convai.utils import (
    get_current_time, 
    format_history_for_llm
)
from convai.data.schemas import (
    SessionCreateResponse,
    ChatMessage,
    MessageResponse,
    MessagesHistoryResponse,
)
from convai.graph import MovieAgentGraph

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # In-Memory Storage
        # TODO: Replace with Database (postgres, mongo or redis)
        self.sessions: Dict[UUID, datetime] = {}
        self.conversations: Dict[UUID, List[ChatMessage]] = {}
        self.agent_graph = MovieAgentGraph()
        
    def create_session(self) -> SessionCreateResponse:
        """
        Create a new chat session.
        """
        logger.info("Creating new chat session")
        session_id = uuid4()
        created_at = get_current_time()
        
        # Store session
        self.sessions[session_id] = created_at
        self.conversations[session_id] = []
        
        logger.info(f"Created new chat session: {session_id}")
        return SessionCreateResponse(
            session_id=session_id,
            created_at=created_at
        )

    def process_message(self, session_id: UUID, message: str) -> MessageResponse:
        """
        Process a user message and generate a response.
        """
        logger.info(f"Received message request for session {session_id}")
        
        # Check if session exists
        if session_id not in self.sessions:
            logger.warning(f"Attempted to send message to non-existent session {session_id}")
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        message_id = uuid4()
        timestamp = get_current_time()
        
        logger.debug(f"Processing user message (ID: {message_id}) for session {session_id}: {message}")
        
        # Retrieve history
        history = self.conversations.get(session_id, [])
        
        logger.debug(f"Retrieved {len(history)} messages from conversation history")
        
        # Format for graph
        conversation_history = format_history_for_llm(history)

        try:
            assistant_response = self.agent_graph.query(message, conversation_history)
            logger.info(f"Successfully generated assistant response for session {session_id}")
            
            # Create assistant message
            assistant_msg_id = uuid4()
            assistant_timestamp = get_current_time()
            
            assistant_msg = ChatMessage(
                message_id=assistant_msg_id,
                role="assistant",
                content=assistant_response,
                timestamp=assistant_timestamp
            )
            
            # Create user message object (we need to store it too)
            user_msg = ChatMessage(
                message_id=message_id,
                role="user",
                content=message,
                timestamp=timestamp
            )
            
            # Save context to memory (list)
            self.conversations[session_id].extend([user_msg, assistant_msg])
            
            # Check current message count for logging
            current_messages = self.conversations[session_id]
            logger.info(f"Successfully processed message for session {session_id}. Total messages: {len(current_messages)}")
            
            return MessageResponse(
                message_id=message_id,
                user_message=message,
                assistant_response=assistant_response,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

    def get_session_history(self, session_id: UUID, limit: int) -> MessagesHistoryResponse:
        """
        Retrieve message history for a specific chat session.
        """
        logger.info(f"Retrieving messages for session {session_id} with limit {limit}")
        
        # Check if session exists
        if session_id not in self.sessions:
            logger.warning(f"Attempted to retrieve messages from non-existent session {session_id}")
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get conversation history from memory
        # Get conversation history
        conversation = self.conversations.get(session_id, [])

        logger.debug(f"Found {len(conversation)} total messages for session {session_id}")
        
        # Apply limit (get most recent messages)
        limited_messages = conversation[-limit:] if len(conversation) > limit else conversation
        
        logger.info(f"Returning {len(limited_messages)} messages for session {session_id}")
        return MessagesHistoryResponse(messages=limited_messages)

# Create a singleton instance
chat_service = ChatService()
