import json
import logging

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from convai.utils.config import settings
from convai.graph.state import GraphState


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize a Tool Calling Agent - which has access to SQL toolkit
        
        Args:
            llm: Language model for Tool calling.
        """
        self.llm = llm
        self.db = SQLDatabase.from_uri(settings.DATABASE_URL)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        self.agent = create_agent(
            model=self.llm, 
            tools=toolkit.get_tools(),
            system_prompt=self._get_agent_system_prompt()
        )
        
        logger.info("Tool Calling Agent initialized")
    

    def _get_agent_system_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_file(f"{settings.PROMPTS_DIR}/tool_agent.prompt")
    

    def generate_and_execute(self, state: GraphState) -> GraphState:
        """
        Makes tool calls (sql tools) and generates and executes SQL queries
        along with user intent and entities extracted in the previous state
        to generate the final response.
        
        Args:
            state: Current graph state with intent and entities
            
        Returns:
            Updated state with final response
        """
        try:
            if not state.get("intent") or not state.get("entities"):
                raise ValueError(
                    "Intent and entities must be extracted before Agent Invocation"
                )
            
            logger.info("Generating SQL query and User Response")
            logger.debug(f"SQL generation context - Intent: {state['intent'].intent.value}, Entities: {state['entities']}")

            result = self.agent.invoke(
                {
                    "messages": [{
                        "role": "user", 
                        "content": f"USER_QUERY: {state['user_query']}\nINTENT: {state['intent'].intent.value}\nENTITIES: {state['entities']}"
                    }]
                }
            )


            print(result)
            res: AIMessage = result["messages"][-1]            
            state["final_response"] = res.content
            
            logger.debug(f"Generated response length: {len(res.content)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Tool Calling Agent error: {e}", exc_info=True)
            state["error"] = f"Tool Calling generation/execution failed: {str(e)}"
            return state
