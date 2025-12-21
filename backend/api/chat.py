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
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Lazy initialization - services will be initialized on first use
mistral_service = None
qdrant_service = None
rag_service = None

def get_mistral_service():
    global mistral_service
    if mistral_service is None:
        try:
            mistral_service = MistralService()
        except Exception as e:
            logger.error(f"Failed to initialize MistralService: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Mistral service unavailable: {str(e)}. Please check MISTRAL_API_KEY environment variable."
            )
    return mistral_service

def get_qdrant_service():
    global qdrant_service
    if qdrant_service is None:
        try:
            qdrant_service = QdrantService()
        except Exception as e:
            logger.error(f"Failed to initialize QdrantService: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Qdrant service unavailable: {str(e)}. Please check QDRANT_URL and QDRANT_API_KEY environment variables."
            )
    return qdrant_service

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService(get_mistral_service(), get_qdrant_service())
    return rag_service

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

        # Get RAG service (lazy initialization)
        rag = get_rag_service()

        # Use RAG service to get response
        result = await rag.process_query(
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

        # Get RAG service (lazy initialization)
        rag = get_rag_service()

        # Index the content using the RAG service
        result = await rag.index_content(
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
    Test connection to all services with detailed information
    """
    result = {
        "mistral": {
            "connected": False,
            "message": "",
            "details": {}
        },
        "qdrant": {
            "connected": False,
            "message": "",
            "details": {}
        },
        "overall": False
    }
    
    # Test Mistral connection
    try:
        mistral = get_mistral_service()
        mistral_connected = await mistral.test_connection()
        result["mistral"]["connected"] = mistral_connected
        
        if mistral_connected:
            # Try to get a test response to verify it's working
            try:
                test_response = await mistral.generate_response("Hello")
                result["mistral"]["message"] = "✓ Mistral AI is connected and responding"
                result["mistral"]["details"]["test_response_length"] = len(test_response)
                result["mistral"]["details"]["model"] = mistral.generation_model
            except Exception as e:
                result["mistral"]["message"] = f"⚠ Mistral connected but test failed: {str(e)[:100]}"
        else:
            result["mistral"]["message"] = "✗ Mistral AI connection test failed"
            
    except HTTPException as e:
        result["mistral"]["message"] = f"✗ {e.detail}"
        result["mistral"]["details"]["error"] = str(e.detail)
    except Exception as e:
        error_msg = str(e)
        result["mistral"]["message"] = f"✗ Mistral AI error: {error_msg[:100]}"
        result["mistral"]["details"]["error"] = error_msg
        if "MISTRAL_API_KEY" in error_msg:
            result["mistral"]["details"]["issue"] = "API key missing or invalid"
    
    # Test Qdrant connection
    try:
        qdrant = get_qdrant_service()
        qdrant_connected = await qdrant.test_connection()
        result["qdrant"]["connected"] = qdrant_connected
        
        if qdrant_connected:
            try:
                point_count = await qdrant.count_points()
                result["qdrant"]["message"] = f"✓ Qdrant is connected ({point_count} documents)"
                result["qdrant"]["details"]["document_count"] = point_count
                result["qdrant"]["details"]["collection"] = qdrant.collection_name
                result["qdrant"]["details"]["url"] = qdrant.url
            except Exception as e:
                result["qdrant"]["message"] = f"⚠ Qdrant connected but count failed: {str(e)[:100]}"
        else:
            result["qdrant"]["message"] = "✗ Qdrant connection test failed"
            
    except HTTPException as e:
        result["qdrant"]["message"] = f"✗ {e.detail}"
        result["qdrant"]["details"]["error"] = str(e.detail)
    except Exception as e:
        error_msg = str(e)
        result["qdrant"]["message"] = f"✗ Qdrant error: {error_msg[:100]}"
        result["qdrant"]["details"]["error"] = error_msg
        if "QDRANT" in error_msg.upper():
            result["qdrant"]["details"]["issue"] = "Qdrant configuration issue"
    
    # Overall status
    result["overall"] = result["mistral"]["connected"] and result["qdrant"]["connected"]
    
    return result