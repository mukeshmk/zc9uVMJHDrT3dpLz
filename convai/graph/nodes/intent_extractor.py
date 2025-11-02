import logging

from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from convai.utils.config import settings
from convai.graph.state import GraphState
from convai.data.schemas import IntentClassification


logger = logging.getLogger(__name__)


class IntentExtractor:    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Intent Classification Node.
        
        Args:
            llm: Language model for intent classification
        """
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{user_query}"),
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(
            IntentClassification
        )
        
        logger.info("Intent Extractor initialized")
    
    def _get_system_prompt(self) -> str:
        template = PromptTemplate.from_file(f"{settings.PROMPTS_DIR}/intent.prompt")
        return template.format()
    
    def classify_intent(self, state: GraphState) -> GraphState:
        """
        Classify user intent from the query.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with intent classification
        """
        try:
            user_query = state["user_query"]
            logger.info(f"Classifying intent for query: {user_query}")
            
            result = self.chain.invoke({
                "user_query": user_query
            })
            
            logger.info(f"Intent classified: {result.intent.value} (confidence: {result.confidence:.2f})")
            logger.debug(f"Intent details: {result}")
            
            state["intent"] = result
            return state
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}", exc_info=True)
            state["error"] = f"Intent classification failed: {str(e)}"
            return state
