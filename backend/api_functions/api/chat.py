from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.openrouter_service import OpenRouterService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize services
openrouter_service = OpenRouterService()
qdrant_service = QdrantService()
rag_service = RAGService(openrouter_service, qdrant_service)

router = APIRouter()

class ChatQuery(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_embedding: Optional[List[float]] = None

class ContentIndexRequest(BaseModel):
    content: str
    chapter_id: str
    section_title: str
    source_url: str

@router.post("/query", response_model=ChatResponse)
async def query_chat(query_data: ChatQuery):
    """
    Process a chat query using RAG (Retrieval Augmented Generation)
    """
    try:
        logger.info(f"Processing query: {query_data.query}")

        # Use RAG service to get response
        result = await rag_service.process_query(
            query=query_data.query,
            user_id=query_data.user_id
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_embedding=result.get("query_embedding")
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index-content")
async def index_content(content_data: ContentIndexRequest):
    """
    Index content for RAG retrieval
    """
    try:
        logger.info(f"Indexing content for chapter: {content_data.chapter_id}")

        # Index the content using the RAG service
        result = await rag_service.index_content(
            content=content_data.content,
            chapter_id=content_data.chapter_id,
            section_title=content_data.section_title,
            source_url=content_data.source_url
        )

        return {"success": True, "document_id": result["document_id"]}
    except Exception as e:
        logger.error(f"Error indexing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-connection")
async def test_connection():
    """
    Test connection to all services
    """
    try:
        # Test OpenRouter connection
        openrouter_status = await openrouter_service.test_connection()

        # Test Qdrant connection
        qdrant_status = await qdrant_service.test_connection()

        return {
            "openrouter": openrouter_status,
            "qdrant": qdrant_status,
            "overall": openrouter_status and qdrant_status
        }
    except Exception as e:
        logger.error(f"Error testing connections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))