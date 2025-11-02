# ConvAI - Conversational AI Movie Recommendation System

A REST API for a conversational AI virtual agent that can answer questions about movies using the MovieLens dataset. The application uses a multi-agent LangGraph workflow to intelligently route, classify, and answer user queries about movies.

## Features

- **Multi-Agent Architecture**: Utilizes LangGraph with specialized agents for routing, intent classification, entity extraction, and SQL generation
- **Natural Language Queries**: Answer questions about movies using natural language
- **MovieLens Dataset**: Works with the MovieLens 100k dataset containing movies, users, ratings, and genres
- **Conversational Context**: Maintains conversation history for context-aware responses
- **RESTful API**: FastAPI-based REST API with comprehensive endpoints
- **Tool Calling**: Calls the SQL tool to automatically generates and executes SQL queries based on user intent

## Architecture

The application uses a LangGraph-based workflow with the following components:

1. **Smart Router**: Determines if the query needs processing or clarification
2. **Intent Extractor**: Classifies user intent (recommendation, specific movie query, genre exploration, etc.)
3. **Entity Extractor**: Extracts structured entities (movie titles, genres, years, ratings) from queries
4. **Tool Calling Agent**: Calls a Tool and generates / executes SQL queries and responds to user query
5. **Error Handler**: Handles errors gracefully throughout the workflow

## Requirements

- Python >= 3.13
- SQLite (included with Python)
- LLM Provider: Ollama (default) or OpenAI
  - For Ollama: Install and run Ollama locally with a model (default: `qwen3:8b`)
  - For OpenAI: Set your API key in environment variables

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

# LLM Configuration
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3:8b
MODEL_TEMPERATURE=0.0

# Logging Configuration
LOG_LEVEL=info
```

For OpenAI instead of Ollama:

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4
# Set OPENAI_API_KEY environment variable or in .env
```

## Running the Application

### Start the Server

```bash
# Run using uv
uv run python convai/app.py

# Or run directly with Python
python -m convai.app

# Or use uvicorn directly
uvicorn convai.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

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

- "Show me action movies from the 1990s"
- "What are the highest rated comedies?"
- "Find movies similar to The Matrix"
- "What movies did user 1 rate highly?"
- "Compare the ratings of Pulp Fiction and Forrest Gump"

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
├── app.py                       # FastAPI application and routes
├── data/
│   ├── database.py              # Database configuration and session management
│   ├── models.py                # SQLAlchemy models (User, Movie, Genre, Rating)
│   ├── schemas.py               # Pydantic schemas for API
│   └── ingest.py                # Data ingestion from MovieLens dataset
├── graph/
│   ├── graph.py                 # Main LangGraph workflow
│   ├── state.py                 # Graph state definition
│   └── nodes/
│       ├── smart_router.py      # Routing agent
│       ├── intent_extractor.py  # Intent classification agent
│       ├── entity_extractor.py  # Entity extraction agent
│       └── agent.py             # Tool calling agent - uses SQLite as a tool and reponds to user query
├── prompts/                     # Prompt templates for LLM agents
├── utils/
│   ├── config.py                # Application configuration
│   ├── download.py              # Dataset download utilities
│   └── logger.py                # Logging configuration
└── tests/                       # Unit tests
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
uvicorn convai.app:app --reload --host 0.0.0.0 --port 8000
```

### Code Quality

The project follows Python best practices and uses:
- FastAPI for the REST API
- SQLite for Database
- SQLAlchemy for database ORM
- LangChain and LangGraph for LLM orchestration
- Pydantic for data validation

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
- If you don't want to use `qwen3:8b` model, make sure to pull and use any other "Tool Calling" model

**OpenAI**:
- Set your API key: `export OPENAI_API_KEY=your_key_here`
- Verify the model name is correct (e.g., `gpt-4`, `gpt-3.5-turbo`)

### Port Already in Use

If port 8000 is already in use, change it:
- Update `PORT` in `.env` file
- Or set environment variable: `export PORT=8001`

## License

This project is provided as-is for demonstration purposes.

## Author

Mukesh Arambakam (amukesh.mk@gmail.com)

