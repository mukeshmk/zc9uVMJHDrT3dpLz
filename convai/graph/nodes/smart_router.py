import logging
from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from convai.utils.config import settings
from convai.graph.state import GraphState
from convai.data.schemas import RouterDecision


logger = logging.getLogger(__name__)
    

class SmartRouter:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Smart Router Node.
        
        Args:
            llm: Language model for Smart Routing Agent's flow
        """
        self.llm = llm

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Conversation History:\n{conversation_history}\n\nUser Query: {user_query}"),
        ])
        
        # Create chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(
            RouterDecision
        )
        
        logger.info("Smart Router initialized")
    
    def _get_system_prompt(self) -> str:
        template = PromptTemplate.from_file(f"{settings.PROMPTS_DIR}/router.prompt")
        return template.format()
    
    def route_query(self, state: GraphState) -> GraphState:
        """
        Extract structured entities from user query.
        
        Args:
            state: Current graph state with intent
            
        Returns:
            Updated state with extracted entities
        """
        try:
            logger.info(f"Routing based on User query: {state["user_query"]}")
            
            # Invoke chain
            result: RouterDecision = self.chain.invoke({
                "user_query": state["user_query"],
                "conversation_history": state["conversation_history"],
            })
            
            # Update state and log
            logger.info(f"Routing decision - Route: {result.route}, Confidence: {result.confidence:.2f}, Reason: {result.reason}")
            state["route"] = result.route
            if result.clarification_message:
                logger.debug(f"Clarification message provided: {result.clarification_message}")
                state["final_response"] = result.clarification_message

            return state
            
        except Exception as e:
            logger.error(f"Route extraction error: {e}", exc_info=True)
            state["error"] = f"Route extraction failed: {str(e)}"
            return state
