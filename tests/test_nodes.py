"""
Unit tests for graph node components.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from convai.graph.nodes import SmartRouter, IntentExtractor, EntityExtractor, Agent
from convai.graph.state import GraphState
from convai.data.schemas import (
    RouterDecision,
    IntentClassification,
    ExtractedEntities,
)


class TestSmartRouter:
    """Tests for SmartRouter node."""
    
    @patch('convai.graph.nodes.smart_router.PromptTemplate')
    def test_router_initialization(self, mock_prompt_template):
        """Test SmartRouter initialization."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_prompt_template.from_file.return_value.format.return_value = "System prompt"
        
        router = SmartRouter(mock_llm)
        
        assert router.llm == mock_llm
        assert router.chain is not None
        mock_llm.with_structured_output.assert_called_once_with(RouterDecision)
    
    def test_route_query_success(self, mock_llm, mock_router_decision):
        """Test successful query routing."""
        # Setup
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(return_value=mock_router_decision)
        
        router = SmartRouter(mock_llm)
        router.chain = mock_chain
        
        state: GraphState = {
            "user_query": "What are the top rated movies?",
            "conversation_history": [],
            "route": None,
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        # Execute
        result = router.route_query(state)
        
        # Verify
        assert result["route"] == "intent_classification"
        assert result.get("error") is None
        mock_chain.invoke.assert_called_once()
    
    def test_route_query_with_clarification_message(self, mock_llm):
        """Test routing with clarification message."""
        clarification_decision = RouterDecision(
            route="ask_clarification",
            confidence=0.8,
            reason="Unclear query",
            clarification_message="Please specify what you're looking for."
        )
        
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(return_value=clarification_decision)
        
        router = SmartRouter(mock_llm)
        router.chain = mock_chain
        
        state: GraphState = {
            "user_query": "Hello",
            "conversation_history": [],
            "route": None,
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = router.route_query(state)
        
        assert result["route"] == "ask_clarification"
        assert result["final_response"] == "Please specify what you're looking for."
    
    def test_route_query_error(self, mock_llm):
        """Test router error handling."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(side_effect=Exception("Routing failed"))
        
        router = SmartRouter(mock_llm)
        router.chain = mock_chain
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": None,
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = router.route_query(state)
        
        assert result.get("error") is not None
        assert "Route extraction failed" in result["error"]


class TestIntentExtractor:
    """Tests for IntentExtractor node."""
    
    @patch('convai.graph.nodes.intent_extractor.PromptTemplate')
    def test_intent_extractor_initialization(self, mock_prompt_template):
        """Test IntentExtractor initialization."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_prompt_template.from_file.return_value.format.return_value = "System prompt"
        
        extractor = IntentExtractor(mock_llm)
        
        assert extractor.llm == mock_llm
        assert extractor.chain is not None
        mock_llm.with_structured_output.assert_called_once_with(IntentClassification)
    
    def test_classify_intent_success(self, mock_llm, mock_intent_classification):
        """Test successful intent classification."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(return_value=mock_intent_classification)
        
        extractor = IntentExtractor(mock_llm)
        extractor.chain = mock_chain
        
        state: GraphState = {
            "user_query": "What are the top rated movies?",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = extractor.classify_intent(state)
        
        assert result["intent"] == mock_intent_classification
        assert result.get("error") is None
        mock_chain.invoke.assert_called_once()
    
    def test_classify_intent_error(self, mock_llm):
        """Test intent classification error handling."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(side_effect=Exception("Classification failed"))
        
        extractor = IntentExtractor(mock_llm)
        extractor.chain = mock_chain
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = extractor.classify_intent(state)
        
        assert result.get("error") is not None
        assert "Intent classification failed" in result["error"]


class TestEntityExtractor:
    """Tests for EntityExtractor node."""
    
    @patch('convai.graph.nodes.entity_extractor.PromptTemplate')
    def test_entity_extractor_initialization(self, mock_prompt_template):
        """Test EntityExtractor initialization."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_prompt_template.from_file.return_value.format.return_value = "System prompt"
        
        extractor = EntityExtractor(mock_llm)
        
        assert extractor.llm == mock_llm
        assert extractor.chain is not None
        mock_llm.with_structured_output.assert_called_once_with(ExtractedEntities)
    
    def test_extract_entities_success(
        self,
        mock_llm,
        mock_intent_classification,
        mock_extracted_entities
    ):
        """Test successful entity extraction."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(return_value=mock_extracted_entities)
        
        extractor = EntityExtractor(mock_llm)
        extractor.chain = mock_chain
        
        state: GraphState = {
            "user_query": "Show me action movies from 1990s",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": mock_intent_classification,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = extractor.extract_entities(state)
        
        assert result["entities"] == mock_extracted_entities
        assert result.get("error") is None
        mock_chain.invoke.assert_called_once()
    
    def test_extract_entities_no_intent_error(self, mock_llm):
        """Test entity extraction fails when intent is missing."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        
        extractor = EntityExtractor(mock_llm)
        
        state: GraphState = {
            "user_query": "Show me action movies",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": None,  # Missing intent
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = extractor.extract_entities(state)
        
        assert result.get("error") is not None
        assert "Intent must be classified" in result["error"]
    
    def test_extract_entities_error(self, mock_llm, mock_intent_classification):
        """Test entity extraction error handling."""
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_chain = MagicMock()
        mock_chain.invoke = Mock(side_effect=Exception("Extraction failed"))
        
        extractor = EntityExtractor(mock_llm)
        extractor.chain = mock_chain
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": mock_intent_classification,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = extractor.extract_entities(state)
        
        assert result.get("error") is not None
        assert "Entity extraction failed" in result["error"]


class TestAgent:
    """Tests for Agent (SQL) node."""
    
    @patch('convai.graph.nodes.agent.SQLDatabase')
    @patch('convai.graph.nodes.agent.SQLDatabaseToolkit')
    @patch('convai.graph.nodes.agent.create_agent')
    @patch('convai.graph.nodes.agent.PromptTemplate')
    def test_agent_initialization(
        self,
        mock_prompt_template,
        mock_create_agent,
        mock_toolkit_class,
        mock_db_class
    ):
        """Test Agent initialization."""
        mock_llm = MagicMock()
        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools = Mock(return_value=[])
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_prompt_template.from_file.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        agent = Agent(mock_llm)
        
        assert agent.llm == mock_llm
        assert agent.db == mock_db
        assert agent.agent is not None
        mock_db_class.from_uri.assert_called_once()
        mock_create_agent.assert_called_once()
    
    @patch('convai.graph.nodes.agent.SQLDatabase')
    @patch('convai.graph.nodes.agent.SQLDatabaseToolkit')
    @patch('convai.graph.nodes.agent.create_agent')
    @patch('convai.graph.nodes.agent.PromptTemplate')
    def test_generate_and_execute_success(
        self,
        mock_prompt_template,
        mock_create_agent,
        mock_toolkit_class,
        mock_db_class,
        mock_llm,
        mock_intent_classification,
        mock_extracted_entities,
        mock_sql_agent_response
    ):
        """Test successful SQL generation and execution."""
        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools = Mock(return_value=[])
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_agent_stream = MagicMock()
        mock_stream = [
            {"step": "thinking"},
            {"model": {"messages": [mock_sql_agent_response]}}
        ]
        mock_agent_stream.stream = Mock(return_value=mock_stream)
        mock_create_agent.return_value = mock_agent_stream
        
        mock_prompt_template.from_file.return_value = MagicMock()
        
        agent = Agent(mock_llm)
        
        state: GraphState = {
            "user_query": "What are the top rated movies?",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": mock_intent_classification,
            "entities": mock_extracted_entities,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = agent.generate_and_execute(state)
        
        assert result["final_response"] == mock_sql_agent_response.content
        assert result.get("error") is None
        mock_agent_stream.stream.assert_called_once()
    
    @patch('convai.graph.nodes.agent.SQLDatabase')
    @patch('convai.graph.nodes.agent.SQLDatabaseToolkit')
    @patch('convai.graph.nodes.agent.create_agent')
    @patch('convai.graph.nodes.agent.PromptTemplate')
    def test_generate_and_execute_no_intent_error(
        self,
        mock_prompt_template,
        mock_create_agent,
        mock_toolkit_class,
        mock_db_class,
        mock_llm
    ):
        """Test SQL generation fails when intent is missing."""
        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools = Mock(return_value=[])
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_create_agent.return_value = MagicMock()
        mock_prompt_template.from_file.return_value = MagicMock()
        
        agent = Agent(mock_llm)
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": None,  # Missing intent
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = agent.generate_and_execute(state)
        
        assert result.get("error") is not None
        assert "Intent and entities must be extracted" in result["error"]
    
    @patch('convai.graph.nodes.agent.SQLDatabase')
    @patch('convai.graph.nodes.agent.SQLDatabaseToolkit')
    @patch('convai.graph.nodes.agent.create_agent')
    @patch('convai.graph.nodes.agent.PromptTemplate')
    def test_generate_and_execute_no_entities_error(
        self,
        mock_prompt_template,
        mock_create_agent,
        mock_toolkit_class,
        mock_db_class,
        mock_llm,
        mock_intent_classification
    ):
        """Test SQL generation fails when entities are missing."""
        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools = Mock(return_value=[])
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_create_agent.return_value = MagicMock()
        mock_prompt_template.from_file.return_value = MagicMock()
        
        agent = Agent(mock_llm)
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": mock_intent_classification,
            "entities": None,  # Missing entities
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = agent.generate_and_execute(state)
        
        assert result.get("error") is not None
        assert "Intent and entities must be extracted" in result["error"]
    
    @patch('convai.graph.nodes.agent.SQLDatabase')
    @patch('convai.graph.nodes.agent.SQLDatabaseToolkit')
    @patch('convai.graph.nodes.agent.create_agent')
    @patch('convai.graph.nodes.agent.PromptTemplate')
    def test_generate_and_execute_error(
        self,
        mock_prompt_template,
        mock_create_agent,
        mock_toolkit_class,
        mock_db_class,
        mock_llm,
        mock_intent_classification,
        mock_extracted_entities
    ):
        """Test SQL generation error handling."""
        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools = Mock(return_value=[])
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_agent_stream = MagicMock()
        mock_agent_stream.stream = Mock(side_effect=Exception("Tool Calling generation failed"))
        mock_create_agent.return_value = mock_agent_stream
        
        mock_prompt_template.from_file.return_value = MagicMock()
        
        agent = Agent(mock_llm)
        
        state: GraphState = {
            "user_query": "Test query",
            "conversation_history": [],
            "route": "intent_classification",
            "intent": mock_intent_classification,
            "entities": mock_extracted_entities,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        result = agent.generate_and_execute(state)
        
        assert result.get("error") is not None
        assert "Tool Calling generation/execution failed" in result["error"]

