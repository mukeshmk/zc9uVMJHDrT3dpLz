import logging

from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from convai.utils.config import settings
from convai.graph.state import GraphState
from convai.data.schemas import ExtractedEntities


logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Entity Extraction Node.
        
        Args:
            llm: Language model for entity extraction
        """
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{user_query}\n\nIntent: {intent}"),
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(
            ExtractedEntities
        )
        
        logger.info("Entity Extractor initialized")
    
    def _get_system_prompt(self) -> str:
        template = PromptTemplate.from_file(f"{settings.PROMPTS_DIR}/entity.prompt")
        return template.format()
    
    def extract_entities(self, state: GraphState) -> GraphState:
        """
        Extract structured entities from user query.
        
        Args:
            state: Current graph state with intent
            
        Returns:
            Updated state with extracted entities
        """
        try:
            user_query = state["user_query"]
            intent = state.get("intent")
            
            if not intent:
                raise ValueError("Intent must be classified before entity extraction")
            
            logger.info(f"Extracting entities from query: {user_query}")
            
            result = self.chain.invoke({
                "user_query": user_query,
                "intent": intent.intent.value
            })
            
            logger.info(f"Entities extracted successfully")
            logger.debug(f"Extracted entities: {result}")
            
            state["entities"] = result
            return state
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}", exc_info=True)
            state["error"] = f"Entity extraction failed: {str(e)}"
            return state
