"""
Unit tests for utility functions.
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from convai.utils import format_history_for_llm
from convai.data.schemas import ChatMessage


def test_format_history_empty_list():
    """Test formatting an empty history list."""
    result = format_history_for_llm([])
    
    assert result == []
    assert isinstance(result, list)

def test_format_history_single_message(sample_chat_message):
    """Test formatting a single message."""
    history = [sample_chat_message]
    result = format_history_for_llm(history)
    
    assert len(result) == 1
    assert result[0]["role"] == sample_chat_message.role
    assert result[0]["content"] == sample_chat_message.content

def test_format_history_multiple_messages():
    """Test formatting multiple messages."""
    messages = [
        ChatMessage(
            message_id=uuid4(),
            role="user",
            content="What are top movies?",
            timestamp=datetime.now(timezone.utc)
        ),
        ChatMessage(
            message_id=uuid4(),
            role="assistant",
            content="Here are some top movies...",
            timestamp=datetime.now(timezone.utc)
        ),
        ChatMessage(
            message_id=uuid4(),
            role="user",
            content="Tell me more",
            timestamp=datetime.now(timezone.utc)
        )
    ]
    
    result = format_history_for_llm(messages)
    
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "What are top movies?"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Here are some top movies..."
    assert result[2]["role"] == "user"
    assert result[2]["content"] == "Tell me more"

def test_format_history_preserves_order():
    """Test that message order is preserved."""
    messages = [
        ChatMessage(
            message_id=uuid4(),
            role="user",
            content=f"Message {i}",
            timestamp=datetime.now(timezone.utc)
        )
        for i in range(5)
    ]
    
    result = format_history_for_llm(messages)
    
    for i, msg in enumerate(result):
        assert msg["content"] == f"Message {i}"

def test_format_history_only_role_and_content(sample_chat_message):
    """Test that only role and content are included in formatted history."""
    history = [sample_chat_message]
    result = format_history_for_llm(history)
    
    formatted_message = result[0]
    
    # Should only have role and content keys
    assert set(formatted_message.keys()) == {"role", "content"}
    assert "message_id" not in formatted_message
    assert "timestamp" not in formatted_message

