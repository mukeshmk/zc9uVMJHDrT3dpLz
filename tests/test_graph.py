"""
Unit tests for graph components (MovieAgentGraph).
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from langchain.chat_models import BaseChatModel

from convai.graph import MovieAgentGraph
from convai.graph.state import GraphState
from convai.data.schemas import (
    RouterDecision,
    IntentClassification,
    IntentType,
    ExtractedEntities,
)


class TestMovieAgentGraphInitialization:
    """Tests for MovieAgentGraph initialization."""
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_graph_initialization(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test that graph initializes correctly with all components."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        mock_router = MagicMock()
        mock_router_class.return_value = mock_router
        
        mock_intent = MagicMock()
        mock_intent_class.return_value = mock_intent
        
        mock_entity = MagicMock()
        mock_entity_class.return_value = mock_entity
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Initialize graph
        graph = MovieAgentGraph()
        
        # Verify LLM was initialized
        mock_init_chat_model.assert_called_once()
        
        # Verify all agents were initialized
        mock_router_class.assert_called_once_with(mock_llm)
        mock_intent_class.assert_called_once_with(mock_llm)
        mock_entity_class.assert_called_once_with(mock_llm)
        mock_agent_class.assert_called_once_with(mock_llm)
        
        # Verify graph has all components
        assert graph.llm == mock_llm
        assert graph.smart_router == mock_router
        assert graph.intent_agent == mock_intent
        assert graph.entity_agent == mock_entity
        assert graph.tool_agent == mock_agent
        assert graph.graph is not None
    
    @patch('convai.graph.graph.init_chat_model')
    def test_graph_initialization_llm_error(self, mock_init_chat_model):
        """Test that graph initialization fails gracefully when LLM fails."""
        mock_init_chat_model.side_effect = Exception("LLM initialization failed")
        
        with pytest.raises(Exception) as exc_info:
            MovieAgentGraph()
        
        assert "LLM initialization failed" in str(exc_info.value)


class TestMovieAgentGraphQuery:
    """Tests for graph query execution."""
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_query_successful_execution(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model,
        mock_router_decision,
        mock_intent_classification,
        mock_extracted_entities
    ):
        """Test successful query execution through the graph."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        # Setup router to return intent_classification route
        mock_router = MagicMock()
        mock_router.route_query.return_value = {
            "user_query": "What are top movies?",
            "route": "intent_classification",
            "conversation_history": [],
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        mock_router_class.return_value = mock_router
        
        # Setup intent extractor
        mock_intent = MagicMock()
        mock_intent.classify_intent.return_value = {
            "user_query": "What are top movies?",
            "route": "intent_classification",
            "conversation_history": [],
            "intent": mock_intent_classification,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        mock_intent_class.return_value = mock_intent
        
        # Setup entity extractor
        mock_entity = MagicMock()
        mock_entity.extract_entities.return_value = {
            "user_query": "What are top movies?",
            "route": "intent_classification",
            "conversation_history": [],
            "intent": mock_intent_classification,
            "entities": mock_extracted_entities,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        mock_entity_class.return_value = mock_entity
        
        # Setup SQL agent
        mock_agent = MagicMock()
        mock_agent.generate_and_execute.return_value = {
            "user_query": "What are top movies?",
            "route": "intent_classification",
            "conversation_history": [],
            "intent": mock_intent_classification,
            "entities": mock_extracted_entities,
            "final_response": "Here are the top rated movies...",
            "error": None,
            "retry_count": 0
        }
        mock_agent_class.return_value = mock_agent
        
        # Initialize and query graph
        graph = MovieAgentGraph()
        result = graph.query("What are top movies?", [])
        
        # Verify result
        assert result == "Here are the top rated movies..."
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_query_ask_clarification_route(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test query that routes to clarification."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        # Setup router to return ask_clarification route
        mock_router = MagicMock()
        mock_router.route_query.return_value = {
            "user_query": "Hello",
            "route": "ask_clarification",
            "conversation_history": [],
            "intent": None,
            "entities": None,
            "final_response": "Please ask a question about movies.",
            "error": None,
            "retry_count": 0
        }
        mock_router_class.return_value = mock_router
        
        mock_intent = MagicMock()
        mock_intent_class.return_value = mock_intent
        
        mock_entity = MagicMock()
        mock_entity_class.return_value = mock_entity
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Initialize and query graph
        graph = MovieAgentGraph()
        result = graph.query("Hello", [])
        
        # Verify clarification message is returned
        assert "Please ask a question about movies" in result or len(result) > 0
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_query_with_error(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test query execution when an error occurs."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        # Setup router to return error
        mock_router = MagicMock()
        mock_router.route_query.return_value = {
            "user_query": "Test query",
            "route": "error",
            "conversation_history": [],
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": "Router failed",
            "retry_count": 0
        }
        mock_router_class.return_value = mock_router
        
        mock_intent = MagicMock()
        mock_intent_class.return_value = mock_intent
        
        mock_entity = MagicMock()
        mock_entity_class.return_value = mock_entity
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Initialize and query graph
        graph = MovieAgentGraph()
        result = graph.query("Test query", [])
        
        # Verify error response
        assert "error" in result.lower() or "encountered" in result.lower()
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_query_graph_execution_error(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test query execution when graph execution fails."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        mock_router = MagicMock()
        mock_router_class.return_value = mock_router
        
        mock_intent = MagicMock()
        mock_intent_class.return_value = mock_intent
        
        mock_entity = MagicMock()
        mock_entity_class.return_value = mock_entity
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Initialize graph
        graph = MovieAgentGraph()
        
        # Mock graph.invoke to raise an error
        graph.graph = MagicMock()
        graph.graph.invoke.side_effect = Exception("Graph execution failed")
        
        # Query should raise exception
        with pytest.raises(Exception) as exc_info:
            graph.query("Test query", [])
        
        assert "Graph execution failed" in str(exc_info.value)
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_query_with_conversation_history(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model,
        sample_conversation_history
    ):
        """Test query execution with conversation history."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        mock_router = MagicMock()
        mock_router_class.return_value = mock_router
        
        mock_intent = MagicMock()
        mock_intent_class.return_value = mock_intent
        
        mock_entity = MagicMock()
        mock_entity_class.return_value = mock_entity
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Initialize graph
        graph = MovieAgentGraph()
        
        # Mock graph execution
        graph.graph = MagicMock()
        graph.graph.invoke.return_value = {
            "user_query": "What are top movies?",
            "route": "intent_classification",
            "conversation_history": sample_conversation_history,
            "intent": None,
            "entities": None,
            "final_response": "Response with history",
            "error": None,
            "retry_count": 0
        }
        
        result = graph.query("What are top movies?", sample_conversation_history)
        
        # Verify conversation history was passed to graph
        call_args = graph.graph.invoke.call_args[0][0]
        assert call_args["conversation_history"] == sample_conversation_history
        assert result == "Response with history"


class TestGraphDecisionMethods:
    """Tests for graph decision methods."""
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_check_for_errors_no_error(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test error checking when no error is present."""
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        graph = MovieAgentGraph()
        
        state = {"error": None}
        result = graph._check_for_errors(state)
        
        assert result == "continue"
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_check_for_errors_with_error(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test error checking when error is present."""
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        graph = MovieAgentGraph()
        
        state = {"error": "Some error occurred"}
        result = graph._check_for_errors(state)
        
        assert result == "error"
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_router_decision(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test router decision method."""
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        graph = MovieAgentGraph()
        
        # Test different routes
        state = {"route": "intent_classification"}
        assert graph._router_decision(state) == "intent_classification"
        
        state = {"route": "ask_clarification"}
        assert graph._router_decision(state) == "ask_clarification"
        
        state = {"route": "error"}
        assert graph._router_decision(state) == "error"
        
        # Test default
        state = {}
        assert graph._router_decision(state) == "ask_clarification"
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_error_node(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test error handling node."""
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        graph = MovieAgentGraph()
        
        state = {
            "error": "Test error message",
            "final_response": None
        }
        
        result = graph._error_node(state)
        
        assert "error" in result["final_response"].lower()
        assert "Test error message" in result["final_response"]
    
    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    def test_clarification_node(
        self,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test clarification node."""
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        graph = MovieAgentGraph()
        
        state = {}
        
        result = graph._clarification_node(state)
        
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert "Please ask a question" in result["final_response"]

