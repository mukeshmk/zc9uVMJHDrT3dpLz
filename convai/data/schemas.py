from uuid import UUID
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

# Request Models

class ChatMessageRequest(BaseModel):
    """
    Request model for sending a message
    """
    message: str = Field(..., min_length=1, description="User message content")


# Response Models

class SessionCreateResponse(BaseModel):
    """
    Response model for session creation
    """
    session_id: UUID = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp in UTC")


class MessageResponse(BaseModel):
    """
    Response model for a single message
    """
    message_id: UUID = Field(..., description="Unique message identifier")
    user_message: str = Field(..., description="Original user message")
    assistant_response: str = Field(..., description="Assistant's response")
    timestamp: datetime = Field(..., description="Message timestamp in UTC")


class ChatMessage(BaseModel):
    """
    Individual message in conversation history
    """
    message_id: UUID = Field(..., description="Unique message identifier")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp in UTC")


class MessagesHistoryResponse(BaseModel):
    """
    Response model for message history
    """
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
