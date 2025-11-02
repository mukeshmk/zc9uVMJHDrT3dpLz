from uuid import uuid4, UUID
from unittest.mock import patch
from fastapi.testclient import TestClient

from convai.app import app
import convai.app as app_module



def test_health_check():
    """Test that health check endpoint returns healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_create_session_success(clean_app_state):
    """Test successful session creation."""
    client = TestClient(app)
    response = client.post("/api/v1/chat/create")
    
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data
    
    # Verify session was stored
    session_id = UUID(data["session_id"])  # Convert string to UUID
    assert session_id in app_module.sessions
    assert session_id in app_module.conversations
    assert len(app_module.conversations[session_id]) == 0


def test_create_multiple_sessions(clean_app_state):
    """Test creating multiple unique sessions."""
    client = TestClient(app)
    
    response1 = client.post("/api/v1/chat/create")
    response2 = client.post("/api/v1/chat/create")
    
    assert response1.status_code == 201
    assert response2.status_code == 201
    
    session_id1 = response1.json()["session_id"]
    session_id2 = response2.json()["session_id"]
    
    assert session_id1 != session_id2
    assert len(app_module.sessions) == 2


def test_send_message_success(clean_app_state, sample_session_id):
    """Test successfully sending a message."""
    # Create a session first
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Mock the graph query
    with patch.object(app_module.agent_graph, 'query', return_value="Mock assistant response"):
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
    
    # Verify messages were stored
    session_id_uuid = UUID(session_id)
    assert len(app_module.conversations[session_id_uuid]) == 2
    assert app_module.conversations[session_id_uuid][0].role == "user"
    assert app_module.conversations[session_id_uuid][1].role == "assistant"


def test_send_message_nonexistent_session(clean_app_state):
    """Test sending message to non-existent session."""
    client = TestClient(app)
    fake_session_id = uuid4()
    
    response = client.post(
        f"/api/v1/chat/{fake_session_id}/messages",
        json={"message": "Hello"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_send_empty_message(clean_app_state, sample_session_id):
    """Test sending an empty message."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.post(
        f"/api/v1/chat/{session_id}/messages",
        json={"message": ""}
    )
    
    assert response.status_code == 422  # Pydantic validation returns 422


def test_send_message_no_request_body(clean_app_state):
    """Test sending message without request body."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.post(
        f"/api/v1/chat/{session_id}/messages",
        json={}
    )
    
    assert response.status_code == 422  # Validation error


def test_send_message_graph_error(clean_app_state):
    """Test handling of graph execution errors."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Mock graph to raise an error
    with patch.object(app_module.agent_graph, 'query', side_effect=Exception("Graph error")):
        response = client.post(
            f"/api/v1/chat/{session_id}/messages",
            json={"message": "Test message"}
        )
    
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()


def test_send_message_with_conversation_history(clean_app_state):
    """Test that conversation history is maintained across messages."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    with patch.object(app_module.agent_graph, 'query', return_value="Response"):
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
    session_id_uuid = UUID(session_id)
    assert len(app_module.conversations[session_id_uuid]) == 4  # 2 user + 2 assistant


def test_get_messages_success(clean_app_state):
    """Test successfully retrieving messages."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Add some messages
    with patch.object(app_module.agent_graph, 'query', return_value="Response"):
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


def test_get_messages_with_limit(clean_app_state):
    """Test retrieving messages with limit parameter."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Add multiple messages
    with patch.object(app_module.agent_graph, 'query', return_value="Response"):
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


def test_get_messages_nonexistent_session(clean_app_state):
    """Test retrieving messages from non-existent session."""
    client = TestClient(app)
    fake_session_id = uuid4()
    
    response = client.get(f"/api/v1/chat/{fake_session_id}/messages")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_messages_empty_conversation(clean_app_state):
    """Test retrieving messages from empty conversation."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    response = client.get(f"/api/v1/chat/{session_id}/messages")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 0


def test_get_messages_limit_validation(clean_app_state):
    """Test limit parameter validation."""
    client = TestClient(app)
    create_response = client.post("/api/v1/chat/create")
    session_id = create_response.json()["session_id"]
    
    # Test limit too low
    response = client.get(f"/api/v1/chat/{session_id}/messages?limit=0")
    assert response.status_code == 422
    
    # Test limit too high
    response = client.get(f"/api/v1/chat/{session_id}/messages?limit=101")
    assert response.status_code == 422

