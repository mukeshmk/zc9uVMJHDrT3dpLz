import logging
from typing import Any, Dict, Optional

from langchain.chat_models import BaseChatModel
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from contextlib import AsyncExitStack

from convai.utils.config import settings
from convai.graph.state import GraphState

logger = logging.getLogger(__name__)


class WeatherAgent:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Weather Agent with MCP tools.
        
        Args:
            llm: Language model for the agent
        """
        self.llm = llm
        self.agent = None
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.mcp_server: str = settings.MCP_SERVER
        
    async def _initialize_agent(self):
        """Initialize the agent with MCP tools asynchronously."""
        if self.agent:
            return

        try:
            if self.mcp_server.startswith("http"):
                read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                    streamablehttp_client(self.mcp_server)
                )
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            elif self.mcp_server.endswith(".py"):
                command = "python" if self.mcp_server.endswith('.py') else "node"
                params = StdioServerParameters(
                    command=command, 
                    args=[self.mcp_server], 
                    env=None
                )
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(params)
                )
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(*stdio_transport)
                )
            else:
                raise ValueError("Must provide either url or server_script_path for MCP_SERVER environment variable.")

            await self.session.initialize()
            
            tools = await load_mcp_tools(
                session=self.session
            )
            
            # Create agent using langchain
            self.agent = create_agent(
                model=self.llm,
                tools=tools,
            )
            
            logger.info("Weather Agent initialized with MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weather Agent: {e}", exc_info=True)
            raise

    async def get_weather(self, state: GraphState) -> GraphState:
        """
        Process weather query.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final response
        """
        try:
            if not self.agent:
                await self._initialize_agent()
                
            logger.info(f"Processing weather query: {state['user_query']}")
            
            # Invoke the agent
            inputs = {"messages": [("user", state["user_query"])]}
            result = await self.agent.ainvoke(inputs)
            
            # Extract the last message content as the response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                state["final_response"] = last_message.content
            else:
                state["final_response"] = "No response from weather agent."
                
            logger.info("Weather query processed successfully")
            
            return state
            
        except Exception as e:
            logger.error(f"Weather Agent error: {e}", exc_info=True)
            state["error"] = f"Weather processing failed: {str(e)}"
            return state
