import logging
from typing import Dict, List, Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START

from convai.utils.config import settings
from convai.graph.state import GraphState
from convai.graph.nodes import IntentExtractor


logger = logging.getLogger(__name__)


class MovieAgentGraph:
    """
    LangGraph workflow orchestrating multiple agents for movie queries.
    """
    
    def __init__(
        self,
        model_provider: str = settings.MODEL_PROVIDER,
        model_name: str = settings.MODEL_NAME,
        temperature: float = settings.MODEL_TEMPERATURE,
    ):
        """
        Initialize the multi-agent graph workflow.
        
        Args:
            model_provider: Model Provider information
            model_name: Which Model Provider's model to use
            temperature: LLM temperature setting
        """

        logger.info(f"Initializing MovieAgentGraph with model={model_name}, provider={model_provider}, temperature={temperature}")
        
        try:
            self.llm = init_chat_model(
                model=model_name, 
                model_provider=model_provider,
                temperature=temperature
            )
            logger.debug(f"LLM initialized successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise
        
        logger.debug("Initializing graph agent")
        self.intent_agent = IntentExtractor(self.llm)
        logger.debug("All agents initialized successfully")
        
        logger.debug("Building graph workflow")
        self.graph = self._build_graph()
        
        logger.info("Movie Agent Graph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        logger.debug("Creating StateGraph builder")

        builder = StateGraph(GraphState)
        
        builder.add_node("intent_classification", self._intent_node)
        builder.add_node("error_handler", self._error_node)
        
        # START -> Smart Router Node
        builder.add_edge(START, "intent_classification")
        
        # Intent -> END (conditional based on errors)
        builder.add_conditional_edges(
            "intent_classification",
            self._check_for_errors,
            {
                "continue": END,
                "error": "error_handler"
            }
        )
        
        # Error handler -> END
        builder.add_edge("error_handler", END)
        
        # Compile graph
        logger.debug("Compiling graph workflow")
        compiled_graph = builder.compile()
        logger.debug("Graph workflow compiled successfully")
        return compiled_graph

    def _intent_node(self, state: GraphState) -> GraphState:
        """Intent Classification node."""
        logger.info("Executing Intent Classification node")
        return self.intent_agent.classify_intent(state)
    
    
    def _error_node(self, state: GraphState) -> GraphState:
        """Error Handling node."""
        logger.error(f"Error occurred: {state.get('error')}")
        
        error_response = f"I encountered an error processing your query: {state.get('error')}"
        state["final_response"] = error_response
        return state
    
    def _check_for_errors(self, state: GraphState) -> Literal["continue", "error"]:
        """
        Check if there are errors in the current state
        
        Args:
            state: Current graph state
            
        Returns:
            "continue" if no errors, "error" if errors present
        """
        if state.get("error"):
            logger.warning(f"Error detected in state: {state.get('error')}")
            return "error"
        logger.debug("No errors detected, continuing workflow")
        return "continue"


    def query(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Process a user query through the multi-agent workflow.
        
        Args:
            user_query: User's movie-related question
            conversation_history: User's previous coversation history with the agent
            
        Returns:
            The final response based on the User's Query and Previous Coversation History
        """

        # Initialize state
        initial_state: GraphState = {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
            "intent": None,
            "final_response": None,
            "error": None,
            "retry_count": 0
        }
        
        logger.info(f"Processing query: {user_query}")
        logger.debug(f"Initial state: intent={initial_state.get('intent')}")
        
        # Execute graph
        try:
            final_state = self.graph.invoke(initial_state)
            logger.debug(f"Graph execution completed.")
        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            raise

        logger.info("Query processing complete")
        
        # Extract final response
        response = "sample response"
        logger.debug(f"Final response length: {len(response)} characters")
        
        return response

