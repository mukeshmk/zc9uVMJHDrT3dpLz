import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from convai.graph import MovieAgentGraph

@pytest.mark.asyncio
class TestWeatherFlow:
    """Tests for weather flow integration."""

    @patch('convai.graph.graph.init_chat_model')
    @patch('convai.graph.graph.SmartRouter')
    @patch('convai.graph.graph.IntentExtractor')
    @patch('convai.graph.graph.EntityExtractor')
    @patch('convai.graph.graph.Agent')
    @patch('convai.graph.graph.WeatherAgent')
    async def test_weather_routing(
        self,
        mock_weather_class,
        mock_agent_class,
        mock_entity_class,
        mock_intent_class,
        mock_router_class,
        mock_init_chat_model
    ):
        """Test that weather queries are routed to the weather agent."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        # Setup router to return weather route
        mock_router = MagicMock()
        mock_router.route_query.return_value = {
            "user_query": "What is the weather in CA?",
            "route": "weather",
            "conversation_history": [],
            "intent": None,
            "entities": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        mock_router_class.return_value = mock_router
        
        # Setup Weather Agent
        mock_weather_agent = AsyncMock()
        mock_weather_agent.get_weather.return_value = {
            "user_query": "What is the weather in CA?",
            "route": "weather",
            "conversation_history": [],
            "intent": None,
            "entities": None,
            "final_response": "Weather in CA is sunny.",
            "error": None,
            "retry_count": 0
        }
        mock_weather_class.return_value = mock_weather_agent
        
        # Initialize graph
        graph = MovieAgentGraph()
        
        # Mock graph execution to verify flow
        # Since we are testing the graph definition, we can check if the router routes to weather
        # But to test the full flow, we need to invoke the graph.
        # However, the graph is compiled with real nodes.
        # We can mock the nodes' methods.
        
        # We need to patch the instance methods on the graph object because the graph is already built
        graph.smart_router = mock_router
        graph.weather_agent = mock_weather_agent
        
        # Execute query
        result = await graph.graph.ainvoke({
            "user_query": "What is the weather in CA?",
            "conversation_history": []
        })
        
        # Verify result
        assert result["final_response"] == "Weather in CA is sunny."
        
        # Verify router was called
        mock_router.route_query.assert_called_once()
        
        # Verify weather agent was called
        mock_weather_agent.get_weather.assert_called_once()

