from enum import Enum
from uuid import UUID
from typing import List, Optional, Literal
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


# LLM Output Formats

class IntentType(str, Enum):
    """
    Enumeration of possible User Intents.
    """
    RECOMMENDATION = "RECOMMENDATION"
    SPECIFIC_MOVIE = "SPECIFIC_MOVIE"
    GENRE_EXPLORATION = "GENRE_EXPLORATION"
    COMPARISON = "COMPARISON"
    TOP_RATED = "TOP_RATED"
    SIMILAR_MOVIES = "SIMILAR_MOVIES"
    GENERAL_QUESTION = "GENERAL_QUESTION"


class RouterDecision(BaseModel):
    """
    Model for router node decision output
    """
    route: Literal["intent_classification", "ask_clarification"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the routing decision (0.0-1.0)")
    reason: str = Field(..., description="Explanation for why this route was chosen")
    clarification_message: str = Field(default="", description="Message to ask user for clarification if routed to ask_clarification")


class IntentClassification(BaseModel):
    """
    Intent classification result from the Intent Agent based on User Query.
    """
    intent: IntentType = Field(..., description="The classified intent type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Brief explanation of why this intent was chosen")


class ExtractedEntities(BaseModel):
    """
    Structured entities extracted from User Query.
    """
    movie_titles: List[str] = Field(default_factory=list, description="List of movie names mentioned in the query")
    genres: List[str] = Field(default_factory=list, description="List of genres mentioned (e.g., Action, Comedy, Drama)")
    year_min: Optional[int] = Field(None, description="Minimum year for movie release date")
    year_max: Optional[int] = Field(None, description="Maximum year for movie release date")
    rating_preference: Optional[str] = Field(None, description="Rating preference description (e.g., 'highly rated', 'top rated')")
    min_rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Minimum rating threshold on 1-5 scale")

