import uvicorn
import logging

from fastapi import FastAPI, status, HTTPException

from convai.utils.config import settings
from convai.utils import get_current_time
from convai.utils.logger import setup_logs
from convai.data.schemas import (
    ChatMessageRequest,
    MessageResponse,
)
from convai.graph.graph import MovieAgentGraph



logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="a REST API for a conversational AI virtual agent that can answer questions \
        about movies using an open movie dataset.",
)

agent_graph = MovieAgentGraph()


def generate_assistant_response(user_message: str) -> str:
    """
    Placeholder for actual AI/LLM integration
    """
    logger.debug(f"Generating assistant response")
    response = agent_graph.query(user_message)
    logger.info(f"Successfully generated assistant response") 
    return response


@app.post(
    "/api/v1/chat/",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
)
async def send_message(
    request: ChatMessageRequest = None
) -> MessageResponse:
    """
    Send a message to the assistant.
    
    Args:
        request: The message request containing the user's message
    
    Returns:
        MessageResponse containing the user message, assistant response, and timestamp
    """
    logger.info(f"Received message request: {request.message}")
    
    timestamp = get_current_time()
    
    # Generate assistant response
    try:
        assistant_response = generate_assistant_response(request.message)
        
        
        return MessageResponse(
            user_message=request.message,
            assistant_response=assistant_response,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy", "timestamp": get_current_time()}


if __name__ == "__main__":
    logger = setup_logs(logger)
    logger.info("Starting Conversational AI FastAPI server")
    logger.info(f"Server configuration: host={settings.HOST}, port={settings.PORT}")
    
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
    )
