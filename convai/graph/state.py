from typing import Dict, List, TypedDict, Optional
from langchain_core.messages import HumanMessage

from convai.data.schemas import IntentClassification


class GraphState(TypedDict):
    """
    State passed between agents in the LangGraph workflow.
    """
    # Input
    user_query: HumanMessage

    # Conversation History
    conversation_history: List[Dict[str, str]]
    
    # Agent outputs
    intent: Optional[IntentClassification]
    final_response: str
    
    # Error handling
    error: Optional[str]
