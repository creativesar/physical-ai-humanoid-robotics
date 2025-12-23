# Physical AI & Humanoid Robotics Textbook - Backend API

This backend API provides RAG (Retrieval Augmented Generation) functionality for the Physical AI & Humanoid Robotics textbook, allowing users to ask questions about the content and receive contextually relevant answers.

## Features

- **RAG Chat Interface**: Question-answering system using textbook content as context
- **Content Indexing**: Automated indexing of textbook content for retrieval
- **Vector Storage**: Qdrant-based vector storage for semantic search
- **Translation**: Multi-language support (Urdu and English)
- **Personalization**: Content adaptation based on user background

## Tech Stack

- **Framework**: FastAPI
- **AI Services**: OpenRouter API (embeddings and generation)
- **Vector Database**: Qdrant
- **Database**: PostgreSQL (Neon)
- **Language**: Python 3.11

## Setup

### Prerequisites

- Python 3.11
- Docker and Docker Compose (for containerized deployment)
- OpenRouter API Key
- Qdrant Cloud account (or local instance)

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```bash
   # Copy the example and fill in your values
   cp .env .env.local
   # Edit .env.local with your actual values
   ```

3. Run the application:
   ```bash
   # Direct execution
   python main.py

   # Or with uvicorn
   uvicorn main:app --reload
   ```

### Docker Setup

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`.

## Content Ingestion

To index the textbook content into the vector database:

1. Make sure your docs are in the `../frontend/docs` directory (or set `DOCS_PATH` environment variable)
2. Run the ingestion script:
   ```bash
   python ingest_content.py
   ```

This will process all markdown files in the docs directory and index them for RAG functionality.

## API Endpoints

### Chat & RAG
- `POST /api/chat/query` - Submit a question and get a RAG-enhanced response
- `POST /api/chat/index-content` - Manually index content for RAG
- `GET /api/chat/test-connection` - Test service connections

### Content Management
- `POST /api/content/index` - Index a single piece of content
- `POST /api/content/batch-index` - Index multiple content items
- `POST /api/content/ingest-from-docs` - Ingest content from docs directory
- `GET /api/content/status` - Get indexing system status

### Translation
- `POST /api/translate/urdu` - Translate content to Urdu
- `POST /api/translate/translate` - Translate between languages
- `GET /api/translate/supported-languages` - Get supported languages

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key for embeddings and generation
- `QDRANT_URL`: URL to your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant (if using cloud)
- `QDRANT_COLLECTION_NAME`: Name of the collection to store vectors (default: "textbook_content")
- `DATABASE_URL`: PostgreSQL connection string (for user data)
- `PORT`: Port to run the API on (default: 8000)

## Architecture

The backend follows a service-oriented architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend APIs   │    │   Vector Store  │
│   (Docusaurus)  │◄──►│   (FastAPI)      │◄──►│   (Qdrant)      │
│                 │    │                  │    │                 │
│ - Chat Widget   │    │ - RAG Service    │    │ - Textbook      │
│ - Auth UI       │    │ - OpenRouter Service │    │   embeddings    │
│ - Translation   │    │ - Qdrant Service │    │ - Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Docker Compose Services

The application is designed to run with Docker Compose alongside the frontend:

- `backend`: FastAPI application
- `db`: PostgreSQL database (Neon)
- `qdrant`: Vector database service

## Development

1. Run tests:
   ```bash
   # Add your test command here
   ```

2. Format code:
   ```bash
   # Add your formatting command here
   ```

## Troubleshooting

- **OpenRouter API errors**: Verify your API key is correct and you have sufficient credits
- **Qdrant connection errors**: Check your Qdrant URL and API key
- **Content not found**: Make sure the docs directory exists and contains markdown files
- **Slow responses**: Check your OpenRouter rate limits and Qdrant performance

## Security

- API keys should never be committed to version control
- Use environment variables for all sensitive configuration
- Implement rate limiting in production
- Validate and sanitize all user inputs