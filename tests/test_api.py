from uuid import uuid4, UUID
from unittest.mock import patch, MagicMock
import pytest

from convai.services.chat import chat_service
from convai.data.repositories.chat_repository import ChatRepository

def test_health_check(client):
    """Test that health check endpoint returns healthy status."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_create_session_success(client, db_session):
    """Test successful session creation."""
    response = client.post("/api/v1/chat/create")
    
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data
    
    # Verify session was stored in DB
    session_id = UUID(data["session_id"])
    repo = ChatRepository(db_session)
    session = repo.get_session(session_id)
    assert session is not None
    assert str(session.session_id) == str(session_id)


def test_create_multiple_sessions(client, db_session):
    """Test creating multiple unique sessions."""
    response1 = client.post("/api/v1/chat/create")
    response2 = client.post("/api/v1/chat/create")
    
    assert response1.status_code == 201
    assert response2.status_code == 201
    
    session_id1 = response1.json()["session_id"]
    session_id2 = response2.json()["session_id"]
    
    assert session_id1 != session_id2
    
    # Verify count in DB
    # We can query directly or use repo (repo doesn't have count method yet, so query directly)
    from convai.data.models import ChatSession
    count = db_session.query(ChatSession).count()
    assert count == 2


def test_send_message_success(client, db_session):
    """Test successfully sending a message."""
    # Create a session first
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Mock the graph query
    with patch.object(chat_service.agent_graph, 'query', return_value="Mock assistant response"):
        response = client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "What are the top rated movies?"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "message_id" in data
    assert data["user_message"] == "What are the top rated movies?"
    assert data["assistant_response"] == "Mock assistant response"
    assert "timestamp" in data
    
    # Verify messages were stored in DB
    repo = ChatRepository(db_session)
    messages = repo.get_messages(UUID(session_id))
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "What are the top rated movies?"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Mock assistant response"


def test_send_message_nonexistent_session(client):
    """Test sending message to non-existent session."""
    fake_session_id = uuid4()
    
    response = client.post(
        f"/api/v1/chat/{fake_session_id}/messages",
        json={"message": "Hello"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_send_empty_message(client):
    """Test sending an empty message."""
    # We need a valid session ID even if validation fails before checking session, 
    # but actually Pydantic validation happens before route handler, so session ID validity doesn't matter 
    # if it's just a path param format check. But let's create one to be safe.
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.post(
        f"/api/v1/chat/{session_id}/messages",
        json={"message": ""}
    )
    
    assert response.status_code == 422  # Pydantic validation returns 422


def test_send_message_no_request_body(client):
    """Test sending message without request body."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.post(
        f"/api/v1/chat/{session_id}/messages",
        json={}
    )
    
    assert response.status_code == 422  # Validation error


def test_send_message_graph_error(client):
    """Test handling of graph execution errors."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Mock graph to raise an error
    with patch.object(chat_service.agent_graph, 'query', side_effect=Exception("Graph error")):
        response = client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "Test message"}
        )
    
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()


def test_send_message_with_conversation_history(client, db_session):
    """Test that conversation history is maintained across messages."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    with patch.object(chat_service.agent_graph, 'query', return_value="Response"):
        # Send first message
        response1 = client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "First message"}
        )
        assert response1.status_code == 200
        
        # Send second message
        response2 = client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "Second message"}
        )
        assert response2.status_code == 200
    
    # Verify all messages are stored
    repo = ChatRepository(db_session)
    messages = repo.get_messages(UUID(session_id))
    assert len(messages) == 4  # 2 user + 2 assistant


def test_get_messages_success(client):
    """Test successfully retrieving messages."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Add some messages
    with patch.object(chat_service.agent_graph, 'query', return_value="Response"):
        client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "Message 1"}
        )
        client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "Message 2"}
        )
    
    # Retrieve messages
    response = client.get(f"/api/v1/chat/{session_id}/messages")
    
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 4  # 2 user + 2 assistant


def test_get_messages_with_limit(client):
    """Test retrieving messages with limit parameter."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Add multiple messages
    with patch.object(chat_service.agent_graph, 'query', return_value="Response"):
        for i in range(5):
            client.post(
                f"/api/v1/chat/{session_id}/messages",
                json={"message": f"Message {i}"}
            )
    
    # Retrieve with limit
    response = client.get(f"/api/v1/chat/{session_id}/messages?limit=2")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 2  # Should return only 2 most recent


def test_get_messages_nonexistent_session(client):
    """Test retrieving messages from non-existent session."""
    fake_session_id = uuid4()
    
    response = client.get(f"/api/v1/chat/{fake_session_id}/messages")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_messages_empty_conversation(client):
    """Test retrieving messages from empty conversation."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.get(f"/api/v1/chat/{session_id}/messages")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 0


def test_get_messages_limit_validation(client):
    """Test limit parameter validation."""
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Test limit too low
    response = client.get(f"/api/v1/chat/{session_id}/messages?limit=0")
    assert response.status_code == 422
    
    # Test limit too high
    response = client.get(f"/api/v1/chat/{session_id}/messages?limit=101")
    assert response.status_code == 422


def test_list_sessions(client, db_session):
    """Test listing sessions via service."""
    # Create a few sessions
    client.post("/api/v1/chat/create")
    client.post("/api/v1/chat/create")
    
    # Use service directly since we don't have an endpoint for listing sessions yet
    # (Streamlit uses service directly)
    sessions = chat_service.list_sessions(db_session)
    assert len(sessions) == 2
    assert isinstance(sessions[0].session_id, UUID)
