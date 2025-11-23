import logging

from typing import List
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from fastapi import HTTPException

from convai.utils import get_current_time, format_history_for_llm
from convai.data.schemas import (
    SessionCreateResponse,
    ChatMessage,
    MessageResponse,
    MessagesHistoryResponse,
)
from convai.graph import MovieAgentGraph
from convai.data.repositories.chat_repository import ChatRepository


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.agent_graph = MovieAgentGraph()

    def create_session(self, db: Session) -> SessionCreateResponse:
        """
        Create a new chat session.
        """
        logger.info("Creating new chat session")
        session_id = uuid4()

        repo = ChatRepository(db)
        session = repo.create_session(session_id)

        logger.info(f"Created new chat session: {session_id}")
        return SessionCreateResponse(
            session_id=UUID(session.session_id), created_at=session.created_at
        )

    async def process_message(
        self, session_id: UUID, message: str, db: Session
    ) -> MessageResponse:
        """
        Process a user message and generate a response.
        """
        logger.info(f"Received message request for session {session_id}")

        repo = ChatRepository(db)
        session = repo.get_session(session_id)

        # Check if session exists
        if not session:
            logger.warning(
                f"Attempted to send message to non-existent session {session_id}"
            )
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        message_id = uuid4()
        timestamp = get_current_time()

        logger.debug(
            f"Processing user message (ID: {message_id}) for session {session_id}: {message}"
        )

        # Retrieve history
        # We fetch recent history for context.
        # TODO: Decide on a reasonable limit for context window or use a summarizer.
        history_msgs = repo.get_messages(session_id, limit=20)

        logger.debug(
            f"Retrieved {len(history_msgs)} messages from conversation history"
        )

        # Format messages for langgraph
        conversation_history = format_history_for_llm(history_msgs)

        try:
            assistant_response = await self.agent_graph.query(
                message, conversation_history
            )
            logger.info(
                f"Successfully generated assistant response for session {session_id}"
            )

            # Save user message
            repo.add_message(
                session_id=session_id,
                role="user",
                content=message,
                message_id=message_id,
            )

            # Save assistant message
            assistant_msg_id = uuid4()
            repo.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_response,
                message_id=assistant_msg_id,
            )

            return MessageResponse(
                message_id=message_id,
                user_message=message,
                assistant_response=assistant_response,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                f"Error processing message for session {session_id}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Error processing message: {str(e)}"
            )

    def get_session_history(
        self, session_id: UUID, limit: int, db: Session
    ) -> MessagesHistoryResponse:
        """
        Retrieve message history for a specific chat session.
        """
        logger.info(f"Retrieving messages for session {session_id} with limit {limit}")

        repo = ChatRepository(db)
        session = repo.get_session(session_id)

        # Check if session exists
        if not session:
            logger.warning(
                f"Attempted to retrieve messages from non-existent session {session_id}"
            )
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        # Get conversation history
        messages = repo.get_messages(session_id, limit=limit)

        logger.info(f"Returning {len(messages)} messages for session {session_id}")

        # Convert DB models to schema models
        pydantic_messages = [
            ChatMessage(
                message_id=UUID(msg.message_id),
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
            )
            for msg in messages
        ]
        return MessagesHistoryResponse(messages=pydantic_messages)

    def list_sessions(self, db: Session) -> List[SessionCreateResponse]:
        """
        List all available chat sessions.
        """
        repo = ChatRepository(db)
        sessions = repo.get_all_sessions()

        return [
            SessionCreateResponse(
                session_id=UUID(s.session_id), created_at=s.created_at
            )
            for s in sessions
        ]


# Create a singleton instance
chat_service = ChatService()
