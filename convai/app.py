import uvicorn
import logging

from typing import List
from uuid import UUID, uuid4
from datetime import datetime
from fastapi import FastAPI, Path, Query, status, HTTPException

from convai.utils.config import settings
from convai.utils.logger import setup_logs
from convai.data.schemas import (
    SessionCreateResponse,
    ChatMessage,
    ChatMessageRequest,
    MessageResponse,
    MessagesHistoryResponse,
)
from convai.graph import MovieAgentGraph
from convai.utils import get_current_time, format_history_for_llm


logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="a REST API for a conversational AI virtual agent that can answer questions \
        about movies using an open movie dataset.",
)

# In-Memory Storage
# TODO: Replace with Database (postgres, mongo or redis)
# Store sessions and conversations (for demonstration purposes)
sessions: dict[UUID, datetime] = {}
conversations: dict[UUID, List[ChatMessage]] = {}
agent_graph = MovieAgentGraph()


def generate_assistant_response(user_message: str, session_id: UUID) -> str:
    """
    Placeholder for actual AI/LLM integration
    Replace this with your movie recommendation logic
    """    
    logger.debug(f"Generating assistant response for session {session_id}")
    conversation_history = conversations.get(session_id, [])
    logger.debug(f"Retrieved {len(conversation_history)} messages from conversation history")
    conversation_history = format_history_for_llm(conversation_history)

    try:
        response = agent_graph.query(user_message, conversation_history)
        logger.info(f"Successfully generated assistant response for session {session_id}")
        return response
    except Exception as e:
        logger.error(f"Error generating assistant response for session {session_id}: {e}", exc_info=True)
        raise


@app.post(
    "/api/v1/chat/create",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session() -> SessionCreateResponse:
    """
    Create a new chat session.
    
    Returns a unique session_id and creation timestamp.
    """
    logger.info("Creating new chat session")
    session_id = uuid4()
    created_at = get_current_time()
    
    # Store session
    sessions[session_id] = created_at
    conversations[session_id] = []
    
    logger.info(f"Created new chat session: {session_id}")
    return SessionCreateResponse(
        session_id=session_id,
        created_at=created_at
    )


@app.post(
    "/api/v1/chat/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
)
async def send_message(
    session_id: UUID = Path(..., description="The session ID"),
    request: ChatMessageRequest = None
) -> MessageResponse:
    """
    Send a message to an existing chat session.
    
    Args:
        session_id: The unique identifier of the chat session
        request: The message request containing the user's message
    
    Returns:
        MessageResponse containing the message ID, user message, 
        assistant response, and timestamp
    """
    logger.info(f"Received message request for session {session_id}")
    
    # Check if session exists
    if session_id not in sessions:
        logger.warning(f"Attempted to send message to non-existent session {session_id}")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    message_id = uuid4()
    timestamp = get_current_time()
    
    logger.debug(f"Processing user message (ID: {message_id}) for session {session_id}: {request.message}")
    
    # Store user message
    user_msg = ChatMessage(
        message_id=message_id,
        role="user",
        content=request.message,
        timestamp=timestamp
    )
    
    # Generate assistant response
    try:
        assistant_response = generate_assistant_response(request.message, session_id)
        assistant_msg_id = uuid4()
        assistant_timestamp = get_current_time()
        
        assistant_msg = ChatMessage(
            message_id=assistant_msg_id,
            role="assistant",
            content=assistant_response,
            timestamp=assistant_timestamp
        )
        
        # Store conversation
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversations[session_id].extend([user_msg, assistant_msg])
        logger.info(f"Successfully processed message for session {session_id}. Total messages: {len(conversations[session_id])}")
        
        return MessageResponse(
            message_id=message_id,
            user_message=request.message,
            assistant_response=assistant_response,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Error processing message for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get(
    "/api/v1/chat/{session_id}/messages",
    response_model=MessagesHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_messages(
    session_id: UUID = Path(..., description="The session ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of messages to return")
) -> MessagesHistoryResponse:
    """
    Retrieve message history for a specific chat session.
    
    Args:
        session_id: The unique identifier of the chat session
        limit: Maximum number of messages to return (default: 10, max: 100)
    
    Returns:
        MessagesHistoryResponse containing the list of messages
    """
    logger.info(f"Retrieving messages for session {session_id} with limit {limit}")
    
    # Check if session exists
    if session_id not in sessions:
        logger.warning(f"Attempted to retrieve messages from non-existent session {session_id}")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get conversation history
    conversation = conversations.get(session_id, [])
    logger.debug(f"Found {len(conversation)} total messages for session {session_id}")
    
    # Apply limit (get most recent messages)
    limited_messages = conversation[-limit:] if len(conversation) > limit else conversation
    
    logger.info(f"Returning {len(limited_messages)} messages for session {session_id}")
    return MessagesHistoryResponse(messages=limited_messages)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy", "timestamp": get_current_time()}


if __name__ == "__main__":
    logger = setup_logs(logger)
    logger.info("Starting Conversational AI FastAPI server")
    logger.info(f"Server configuration: host={settings.HOST}, port={settings.PORT}")
    
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        log_config=None,
    )
