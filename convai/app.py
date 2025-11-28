import uvicorn
import logging

from uuid import UUID
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from fastapi import FastAPI, Path, Query, status, Depends

from convai.utils.config import settings
from convai.utils.logger import setup_logs
from convai.data.schemas import (
    SessionCreateResponse,
    ChatMessageRequest,
    MessageResponse,
    MessagesHistoryResponse,
)
from convai.utils import get_current_time
from convai.services.chat import chat_service
from convai.data.database import get_db, init_db


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown logic.
    """
    logger.info("Initializing database...")
    init_db()
    yield
    # Shutdown logic can go here if needed


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="a REST API for a conversational AI virtual agent that can answer questions \
        about movies using an open movie dataset.",
    lifespan=lifespan,
)


@app.post(
    "/api/v1/chat/create",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(db: Session = Depends(get_db)) -> SessionCreateResponse:
    """
    Creates a new chat session.

    Returns a unique session_id and creation timestamp.
    """
    return chat_service.create_session(db)


@app.post(
    "/api/v1/chat/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
)
async def send_message(
    session_id: UUID = Path(..., description="The session ID"),
    request: ChatMessageRequest = None,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Sends a message to an existing chat session.

    Args:
        session_id: The unique identifier of the chat session
        request: The message request containing the user's message

    Returns:
        MessageResponse containing the message ID, user message,
        assistant response, and timestamp
    """
    return await chat_service.process_message(session_id, request.message, db)


@app.get(
    "/api/v1/chat/{session_id}/messages",
    response_model=MessagesHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_messages(
    session_id: UUID = Path(..., description="The session ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of messages to return"),
    db: Session = Depends(get_db),
) -> MessagesHistoryResponse:
    """
    Retrieves message history for a specific chat session.

    Args:
        session_id: The unique identifier of the chat session
        limit: Maximum number of messages to return (default: 10, max: 100)

    Returns:
        MessagesHistoryResponse containing the list of messages
    """
    return chat_service.get_session_history(session_id, limit, db)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
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
