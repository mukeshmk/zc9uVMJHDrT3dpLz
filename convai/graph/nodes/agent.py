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

        def custom_prompt(request: ModelRequest) -> str:
            intent = request.runtime.context.get("intent", None)
            entities = request.runtime.context.get("entities", None)
            user_query = request.runtime.context.get("user_query", None)
            conversation_history = request.runtime.context.get("conversation_history", None)

            system_message = self._get_agent_system_prompt()
            prompt = system_message.format(
                dialect=self.db.dialect, 
                top_k=5,
                intent=intent,
                entities=entities,
                user_query=user_query,
                conversation_history=conversation_history,
            )
            return prompt
        
        custom_prompt = dynamic_prompt(custom_prompt)

        self.agent = create_agent(
            model=self.llm, 
            tools=toolkit.get_tools(),
            middleware=[
                custom_prompt
            ]
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

            result = []
            step_count = 0
            for step in self.agent.stream(
                {
                    "messages": [{
                        "role": "user", 
                        "content": state["user_query"]
                    }]
                },
                context={
                    "intent": state['intent'].intent.value,
                    "entities": state["entities"],
                    "user_query": state["user_query"],
                    "conversation_history": state["conversation_history"],
                }
            ):
                result.append(step)
                step_count += 1
                logger.debug(f"Tool Calling Agent step {step_count}: {json.dumps(step, indent=2, default=str)}")

            res: AIMessage = result[-1]["model"]["messages"][-1]            
            state["final_response"] = res.content
            
            logger.info(f"Tool Calling Agent completed successfully after {step_count} steps")
            logger.debug(f"Generated response length: {len(res.content)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Tool Calling Agent error: {e}", exc_info=True)
            state["error"] = f"Tool Calling generation/execution failed: {str(e)}"
            return state
