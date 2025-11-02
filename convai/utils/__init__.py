from typing import Dict, List
from datetime import datetime, timezone

from convai.data.schemas import ChatMessage


def get_current_time() -> datetime:
    """
    Get current time
    """
    return datetime.now(timezone.utc)


def format_history_for_llm(history: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Convert ChatMessage objects to LLM-compatible format
    """
    formatted_history = []
    for msg in history:
        formatted_history.append({
            "role": msg.role,
            "content": msg.content
        })
    return formatted_history


__all__ = [
    "get_current_time",
    "format_history_for_llm",
]
