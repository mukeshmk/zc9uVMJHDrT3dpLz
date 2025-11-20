import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import desc

from convai.data.models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)

class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, session_id: UUID) -> ChatSession:
        """Create a new chat session."""
        db_session = ChatSession(
            session_id=str(session_id),
            created_at=datetime.now(),
            last_active_at=datetime.now()
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def get_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.db.query(ChatSession).filter(ChatSession.session_id == str(session_id)).first()

    def get_all_sessions(self) -> List[ChatSession]:
        """Get all chat sessions ordered by last active time."""
        return self.db.query(ChatSession).order_by(desc(ChatSession.last_active_at)).all()

    def add_message(self, session_id: UUID, role: str, content: str, message_id: UUID) -> ChatMessage:
        """Add a message to a session."""
        db_message = ChatMessage(
            message_id=str(message_id),
            session_id=str(session_id),
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.db.add(db_message)
        
        # Update session last active time
        session = self.get_session(session_id)
        if session:
            session.last_active_at = datetime.now()
            self.db.add(session)
            
        self.db.commit()
        self.db.refresh(db_message)
        return db_message

    def get_messages(self, session_id: UUID, limit: int = None) -> List[ChatMessage]:
        """Get messages for a session."""
        query = self.db.query(ChatMessage).filter(ChatMessage.session_id == str(session_id))
        query = query.order_by(ChatMessage.timestamp.asc())
        
        if limit:
            # To get the *last* N messages, we need to be careful.
            # If we just limit, we get the first N.
            # If we order desc and limit, we get the last N but in reverse order.
            # So we get total count or do a subquery, or just fetch all and slice (if not too many).
            # Given chat history isn't massive, fetching all and slicing is okay for now, 
            # but better to do it in DB.
            
            total_count = query.count()
            if total_count > limit:
                query = query.offset(total_count - limit)
                
        return query.all()
