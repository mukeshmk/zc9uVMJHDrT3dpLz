from datetime import datetime
from pydantic import BaseModel, Field


# Request Models

class ChatMessageRequest(BaseModel):
    """
    Request model for sending a message
    """
    message: str = Field(..., min_length=1, description="User message content")


# Response Models

class MessageResponse(BaseModel):
    """
    Response model for a single message
    """
    user_message: str = Field(..., description="Original user message")
    assistant_response: str = Field(..., description="Assistant's response")
    timestamp: datetime = Field(..., description="Message timestamp in UTC")
