import pytest

from unittest.mock import Mock, MagicMock
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from convai.app import app
from convai.data.schemas import (
    ChatMessage,
    IntentClassification,
    IntentType,
    ExtractedEntities,
    RouterDecision,
)
from convai.data.database import Base, get_db
from convai.data.repositories.chat_repository import ChatRepository

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID for testing."""
    return uuid4()


@pytest.fixture
def sample_timestamp():
    """Generate a sample timestamp for testing."""
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_chat_message(sample_session_id, sample_timestamp):
    """Create a sample chat message."""
    return ChatMessage(
        message_id=sample_session_id,
        role="user",
        content="What are the top rated movies?",
        timestamp=sample_timestamp
    )


@pytest.fixture
def sample_conversation_history():
    """Create a sample conversation history."""
    return [
        {
            "role": "user",
            "content": "What are the top rated movies?"
        },
        {
            "role": "assistant",
            "content": "Here are some top rated movies..."
        }
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_router_decision():
    """Create a mock router decision."""
    return RouterDecision(
        route="intent_classification",
        confidence=0.95,
        reason="Query is clear and movie-related",
        clarification_message=""
    )


@pytest.fixture
def mock_intent_classification():
    """Create a mock intent classification."""
    return IntentClassification(
        intent=IntentType.TOP_RATED,
        confidence=0.9,
        reasoning="User is asking about top rated movies"
    )


@pytest.fixture
def mock_extracted_entities():
    """Create mock extracted entities."""
    return ExtractedEntities(
        movie_titles=[],
        genres=[],
        year_min=None,
        year_max=None,
        rating_preference="highly rated",
        min_rating=4.0
    )


@pytest.fixture
def mock_graph_state():
    """Create a mock graph state."""
    return {
        "user_query": "What are the top rated movies?",
        "route": None,
        "conversation_history": [],
        "intent": None,
        "entities": None,
        "final_response": None,
        "error": None,
        "retry_count": 0
    }


@pytest.fixture
def mock_sql_agent_response():
    """Create a mock SQL agent response."""
    from langchain_core.messages import AIMessage
    return AIMessage(content="Based on the database, here are the top rated movies: The Shawshank Redemption, The Godfather...")


@pytest.fixture
def mock_compiled_graph():
    """Create a mock compiled graph."""
    graph = MagicMock()
    
    def invoke_side_effect(state):
        """Simulate graph execution."""
        state = state.copy()
        if state.get("error"):
            return state
        state["route"] = "intent_classification"
        state["final_response"] = "Mock response from graph"
        return state
    
    graph.invoke = Mock(side_effect=invoke_side_effect)
    return graph


@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    db = MagicMock()
    db.dialect = "sqlite"
    return db


@pytest.fixture
def mock_toolkit():
    """Create a mock SQL toolkit."""
    toolkit = MagicMock()
    toolkit.get_tools = Mock(return_value=[])
    return toolkit

@pytest.fixture(scope="function")
def db_session():
    """
    Create a fresh database session for a test.
    """
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    """
    Create a TestClient that uses the test database session.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def sample_session_id():
    """Generate a sample session ID for testing."""
    return uuid4()


@pytest.fixture
def sample_timestamp():
    """Generate a sample timestamp for testing."""
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_chat_message(sample_session_id, sample_timestamp):
    """Create a sample chat message."""
    return ChatMessage(
        message_id=sample_session_id,
        role="user",
        content="What are the top rated movies?",
        timestamp=sample_timestamp
    )


@pytest.fixture
def sample_conversation_history():
    """Create a sample conversation history."""
    return [
        {
            "role": "user",
            "content": "What are the top rated movies?"
        },
        {
            "role": "assistant",
            "content": "Here are some top rated movies..."
        }
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_router_decision():
    """Create a mock router decision."""
    return RouterDecision(
        route="intent_classification",
        confidence=0.95,
        reason="Query is clear and movie-related",
        clarification_message=""
    )


@pytest.fixture
def mock_intent_classification():
    """Create a mock intent classification."""
    return IntentClassification(
        intent=IntentType.TOP_RATED,
        confidence=0.9,
        reasoning="User is asking about top rated movies"
    )


@pytest.fixture
def mock_extracted_entities():
    """Create mock extracted entities."""
    return ExtractedEntities(
        movie_titles=[],
        genres=[],
        year_min=None,
        year_max=None,
        rating_preference="highly rated",
        min_rating=4.0
    )


@pytest.fixture
def mock_graph_state():
    """Create a mock graph state."""
    return {
        "user_query": "What are the top rated movies?",
        "route": None,
        "conversation_history": [],
        "intent": None,
        "entities": None,
        "final_response": None,
        "error": None,
        "retry_count": 0
    }


@pytest.fixture
def mock_sql_agent_response():
    """Create a mock SQL agent response."""
    from langchain_core.messages import AIMessage
    return AIMessage(content="Based on the database, here are the top rated movies: The Shawshank Redemption, The Godfather...")


@pytest.fixture
def mock_compiled_graph():
    """Create a mock compiled graph."""
    graph = MagicMock()
    
    def invoke_side_effect(state):
        """Simulate graph execution."""
        state = state.copy()
        if state.get("error"):
            return state
        state["route"] = "intent_classification"
        state["final_response"] = "Mock response from graph"
        return state
    
    graph.invoke = Mock(side_effect=invoke_side_effect)
    return graph


@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    db = MagicMock()
    db.dialect = "sqlite"
    return db


@pytest.fixture
def mock_toolkit():
    """Create a mock SQL toolkit."""
    toolkit = MagicMock()
    toolkit.get_tools = Mock(return_value=[])
    return toolkit

