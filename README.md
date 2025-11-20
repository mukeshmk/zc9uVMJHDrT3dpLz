# ConvAI - Conversational AI Movie Recommendation System

A REST API for a conversational AI virtual agent that can answer questions about movies using the MovieLens dataset. The application uses a multi-agent LangGraph workflow to intelligently route, classify, and answer user queries about movies.

## Features

- **Multi-Agent Architecture**: Utilizes LangGraph with specialized agents for routing, intent classification, entity extraction, SQL generation, and weather queries
- **Service Layer Architecture**: Clean separation of business logic with dedicated `ChatService` layer for session and conversation management
- **Natural Language Queries**: Answer questions about movies and weather using natural language
- **MovieLens Dataset**: Works with the MovieLens 100k dataset containing movies, users, ratings, and genres
- **MCP Server Integration**: Extensible Model Context Protocol (MCP) server for weather data with both HTTP and stdio transport support
- **Weather Agent**: Dedicated agent for weather forecasts and alerts using the National Weather Service API
- **Streamlit Web UI**: Interactive web-based chat interface for seamless user interaction
- **Conversational Context**: Maintains conversation history for context-aware responses with persistent SQLite storage
- **Asynchronous Processing**: Fully async graph execution and message processing for improved performance
- **RESTful API**: FastAPI-based REST API with comprehensive endpoints
- **Multiple LLM Providers**: Support for Ollama (local), OpenAI, and Groq inference models
- **Tool Calling**: Automatically generates and executes SQL queries and weather API calls based on user intent

## Architecture

The application uses a **service-oriented, multi-agent LangGraph architecture** with the following components:

### Core Workflow Agents

1. **Smart Router**: Determines if the query is about movies, weather, or needs clarification
2. **Intent Extractor**: Classifies user intent (recommendation, specific movie query, genre exploration, weather forecast, etc.)
3. **Entity Extractor**: Extracts structured entities (movie titles, genres, years, ratings, locations) from queries
4. **Tool Calling Agent**: Generates and executes SQL queries for movie data and responds to user queries
5. **Weather Agent**: Processes weather-related queries using the MCP Server to fetch forecasts and alerts
6. **Error Handler**: Handles errors gracefully throughout the workflow

### Service Layer

- **ChatService**: Business logic layer that manages:
  - Session creation and tracking
  - Conversation history (persistent SQLite storage)
  - Message processing and coordination with the agent graph
  - Response generation

### MCP Server Integration

- **Weather MCP Server**: Model Context Protocol server providing:
  - Weather forecast tool (using latitude/longitude)
  - Weather alerts tool (using US state codes)
  - Powered by the National Weather Service API
  - Supports both HTTP and stdio transport protocols

## Requirements

- Python >= 3.13
- SQLite (included with Python)
- LLM Provider (choose one):
  - **Ollama** (default): Run locally with tool-calling compatible models (e.g., `qwen3:8b`)
  - **OpenAI**: GPT-4, GPT-3.5-turbo, or other OpenAI models
  - **Groq**: Fast inference with models like `llama-4-scout-17b-16e-instruct`
- **Streamlit** (optional): For the web-based chat UI
- **MCP Server Dependencies** (optional): For weather functionality via Model Context Protocol

## Installation

### Option 1: Using `uv` (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. If you don't have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or install the package in editable mode
uv pip install -e .
```

### Option 2: Using `pip`

If you prefer using `pip`, you can install the dependencies with:

```bash
# Install dependencies
pip install -r requirements.txt

# Or install the package in editable mode
pip install -e .
```

## Setup

### 1. Install LLM Provider (Ollama - Default)

If using Ollama (the default provider), install and set it up:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (default: qwen3:8b)
ollama pull qwen3:8b
```

Or use any other Tool Calling compatible model from Ollama.

### 2. Initialize the Database

The application uses SQLite and automatically downloads the MovieLens 100k dataset on first run. To initialize the database:

```bash
# Run the data ingestion script
python -m convai.data.ingest
```

This will:
- Download the MovieLens 100k dataset
- Extract it to a temporary directory
- Load users, movies, genres, and ratings into the database
- Create `movielens.db` in the project root
- And cleans up after the download, extraction and database creattion

### 3. Configure Environment Variables (Optional)

Create a `.env` file in the project root to customize settings:

```env
# API Configuration
HOST=0.0.0.0
PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./movielens.db

# LLM Configuration (Ollama - Default)
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:8b
MODEL_TEMPERATURE=0.0

# MCP Server Configuration (for Weather functionality)
MCP_SERVER=http://127.0.0.1:8001/mcp

# Logging Configuration
LOG_LEVEL=info
```

**For OpenAI:**

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4
API_KEY=your_openai_api_key_here
```

**For Groq (Fast Inference):**

```env
MODEL_PROVIDER=groq
MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
API_KEY=your_groq_api_key_here
```

### 4. Start the MCP Weather Server (Optional)

To enable weather functionality, start the MCP server:

```bash
# Start with HTTP transport (default port 8001)
uv run python mcp_server/weather_server.py --transport http

# Or start with stdio transport
uv run python mcp_server/weather_server.py --transport stdio
```

The MCP server provides:
- **Weather Forecasts**: Get detailed forecasts using latitude/longitude
- **Weather Alerts**: Check active weather alerts by US state code

Leave this running in a separate terminal if you want weather query support.

## Running the Application

### Option 1: FastAPI Server (REST API)

Start the FastAPI server for REST API access:

```bash
# Run using uv
uv run python convai/app.py

# Or run directly with Python
python -m convai.app

# Or use uvicorn directly
uvicorn convai.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Option 2: Streamlit Web UI (Interactive Chat)

Start the Streamlit web interface for an interactive chat experience:

```bash
# Run from project root
streamlit run convai/ui/streamlit_app.py

# Or specify a custom port
streamlit run convai/ui/streamlit_app.py --server.port 8502
```

The web UI will open in your browser at `http://localhost:8501`.

**Features:**
- 🆕 Create new chat sessions with one click
- 💬 Manage multiple conversations
- 🔄 Switch between sessions seamlessly
- 📜 View complete conversation history
- 🎨 Modern, intuitive interface

### Running Both (Recommended for Full Experience)

For the complete experience with both API and UI:

```bash
# Terminal 1: Start MCP Weather Server (optional, for weather queries)
uv run python mcp_server/weather_server.py --transport http

# Terminal 2: Start FastAPI Server (if you want API access)
uv run python convai/app.py

# Terminal 3: Start Streamlit UI
streamlit run convai/ui/streamlit_app.py
```

### API Documentation

Once the server is running, you can access:
- **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

## Usage

### 1. Create a Chat Session

```bash
curl -X POST http://localhost:8000/api/v1/chat/create
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 2. Send a Message

```bash
curl -X POST http://localhost:8000/api/v1/chat/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the top 5 rated action movies?"
  }'
```

Response:
```json
{
  "message_id": "660e8400-e29b-41d4-a716-446655440001",
  "user_message": "What are the top 5 rated action movies?",
  "assistant_response": "Here are the top 5 rated action movies:\n1. The Shawshank Redemption (1994) - 4.8\n2. The Godfather (1972) - 4.8\n...",
  "timestamp": "2024-01-15T10:30:05Z"
}
```

### 3. Get Conversation History

```bash
curl http://localhost:8000/api/v1/chat/{session_id}/messages?limit=10
```

### Example Queries

**Movie Queries:**
- "Show me action movies from the 1990s"
- "What are the highest rated comedies?"
- "Find movies similar to The Matrix"
- "What movies did user 1 rate highly?"
- "Compare the ratings of Pulp Fiction and Forrest Gump"

**Weather Queries:**
- "What's the weather forecast for San Francisco?" (uses lat/long: 37.7749, -122.4194)
- "Are there any weather alerts in California?" (state code: CA)
- "Show me the weather for New York City" (uses lat/long: 40.7128, -74.0060)
- "Get weather alerts for Texas" (state code: TX)

## API Endpoints

### `POST /api/v1/chat/create`
Create a new chat session.

**Response**: Session ID and creation timestamp

### `POST /api/v1/chat/{session_id}/messages`
Send a message to an existing session.

**Request Body**:
```json
{
  "message": "Your question about movies"
}
```

**Response**: Message ID, user message, assistant response, and timestamp

### `GET /api/v1/chat/{session_id}/messages`
Retrieve message history for a session.

**Query Parameters**:
- `limit` (optional): Number of messages to return (default: 10, max: 100)

**Response**: List of messages in the conversation

### `GET /health`
Health check endpoint.

**Response**: Service status and timestamp

## Project Structure

```
convai/
├── app.py                       # FastAPI application and API routes
├── data/
│   ├── database.py              # Database configuration and session management
│   ├── models.py                # SQLAlchemy models (User, Movie, Genre, Rating)
│   ├── schemas.py               # Pydantic schemas for API requests/responses
│   └── ingest.py                # Data ingestion from MovieLens dataset
├── services/
│   └── chat.py                  # ChatService - business logic layer for session & conversation management
├── graph/
│   ├── graph.py                 # Main LangGraph workflow orchestration
│   ├── state.py                 # Graph state definition
│   └── nodes/
│       ├── smart_router.py      # Routing agent (movies vs weather vs clarification)
│       ├── intent_extractor.py  # Intent classification agent
│       ├── entity_extractor.py  # Entity extraction agent
│       ├── agent.py             # Tool calling agent - SQL query generation and execution
│       └── weather_agent.py     # Weather agent - MCP-based weather queries
├── ui/
│   ├── streamlit_app.py         # Streamlit web interface for interactive chat
│   └── README.md                # Streamlit UI documentation
├── prompts/                     # Prompt templates for LLM agents (.prompt files)
├── utils/
│   ├── config.py                # Application configuration and settings
│   ├── download.py              # Dataset download utilities
│   └── logger.py                # Logging configuration
└── tests/                       # Unit and integration tests
    ├── test_api.py              # FastAPI endpoint tests
    ├── test_graph.py            # LangGraph workflow tests
    └── test_weather_flow.py     # Weather agent integration tests

mcp_server/
└── weather_server.py            # MCP Weather Server (HTTP/stdio transport)
```

## Testing

Run the test suite:

```bash
# Using pytest
pytest tests/

# With coverage
pytest tests / --cov=convai --cov-report=html
```

## Development

### Running in Development Mode

For development with auto-reload:

```bash
# FastAPI with auto-reload
uvicorn convai.app:app --reload --host 0.0.0.0 --port 8000

# Streamlit with auto-reload (default behavior)
streamlit run convai/ui/streamlit_app.py
```

### Code Quality

The project follows Python best practices and uses:
- **FastAPI** for the REST API
- **Streamlit** for the web UI
- **SQLite** for database storage
- **SQLAlchemy** for database ORM
- **LangChain** and **LangGraph** for LLM orchestration
- **MCP (Model Context Protocol)** for extensible tool integration
- **Pydantic** for data validation
- **Asyncio** for asynchronous processing

### Testing

The test suite includes:
- API endpoint tests (`test_api.py`)
- Graph workflow tests (`test_graph.py`) with async support
- Weather agent integration tests (`test_weather_flow.py`)

## Troubleshooting

### Database Issues

If you encounter database errors:
1. Ensure the database file `movielens.db` exists
2. Re-run the ingestion script (after deleting `movielens.db` if it exists): `python -m convai.data.ingest`

### LLM Provider Issues

**Ollama**:
- Ensure Ollama is running: `ollama serve`
- Verify the model is available: `ollama list`
- Pull the model if missing: `ollama pull qwen3:8b`
- If you don't want to use `qwen3:8b` model, make sure to pull and use any other "Tool Calling" compatible model

**OpenAI**:
- Set your API key: `export OPENAI_API_KEY=your_key_here` or add `API_KEY` to `.env`
- Verify the model name is correct (e.g., `gpt-4`, `gpt-3.5-turbo`)

**Groq**:
- Get your API key from [Groq Console](https://console.groq.com/)
- Set in `.env`: `API_KEY=your_groq_api_key_here`
- Ensure `MODEL_PROVIDER=groq` is set
- Verify model name matches available Groq models

### MCP Server Issues

**Weather queries not working**:
- Ensure MCP server is running: `uv run python mcp_server/weather_server.py --transport http`
- Check `MCP_SERVER` in `.env` matches the server URL (default: `http://127.0.0.1:8001/mcp`)
- Verify port 8001 is not in use by another process

**Connection errors**:
- For HTTP transport, ensure the server URL is correct
- For stdio transport, ensure Python is in your PATH

### Streamlit UI Issues

**Port already in use**:
```bash
streamlit run convai/ui/streamlit_app.py --server.port 8502
```

**Import errors**:
- Ensure you're in the project root directory
- Activate virtual environment: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Async loop errors**:
- Ensure Python 3.8+ is installed
- Update Streamlit: `pip install --upgrade streamlit`

### Port Already in Use

If default ports are already in use, change them:

**FastAPI (default 8000)**:
- Update `PORT` in `.env` file
- Or: `export PORT=8001`

**MCP Server (default 8001)**:
- Update `MCP_SERVER` in `.env`
- Modify `weather_server.py` port configuration

**Streamlit (default 8501)**:
- Use `--server.port` flag: `streamlit run convai/ui/streamlit_app.py --server.port 8502`

## License

This project is provided as-is for demonstration purposes.

## Author

Mukesh Arambakam (amukesh.mk@gmail.com)

