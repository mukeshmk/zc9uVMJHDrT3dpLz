import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from convai.utils.config import settings


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
    
    def query(self, user_query: str) -> str:
        """
        Process a user query through the multi-agent workflow.
        
        Args:
            user_query: User's movie-related question
            
        Returns:
            The final response
        """

        prompt = """
You are a helpful and friendly conversational AI assistant. Your goal is to provide accurate, relevant, and concise responses to user queries.

Guidelines:
- Be conversational and natural in your responses
- Ask clarifying questions when the user's intent is unclear
- Admit when you don't know something rather than making up information
- Keep responses focused and avoid unnecessary verbosity
- Maintain context from previous messages in the conversation
- Be respectful and professional at all times

Respond directly to the user's question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("user", f"USER_QUERY: {user_query}")
        ])

        # Create chain with structured output
        self.chain = prompt | self.llm

        # Invoke chain
        result = self.chain.invoke({
            "user_query": user_query
        })

        return result.content

