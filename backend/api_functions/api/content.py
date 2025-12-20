from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
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

# Initialize services
mistral_service = MistralService()
qdrant_service = QdrantService()
rag_service = RAGService(mistral_service, qdrant_service)

router = APIRouter()

class ContentIndexRequest(BaseModel):
    content: str
    chapter_id: str
    section_title: str
    source_url: str

class BatchContentIndexRequest(BaseModel):
    contents: List[ContentIndexRequest]

class PersonalizeContentRequest(BaseModel):
    chapter_id: str
    user_id: str
    user_background: Dict[str, Any]

@router.post("/index")
async def index_content(content_data: ContentIndexRequest):
    """
    Index a single piece of content for RAG retrieval
    """
    try:
        logger.info(f"Indexing content for chapter: {content_data.chapter_id}")

        result = await rag_service.index_content(
            content=content_data.content,
            chapter_id=content_data.chapter_id,
            section_title=content_data.section_title,
            source_url=content_data.source_url
        )

        return {
            "success": True,
            "document_id": result["document_id"],
            "indexed_content_length": result["indexed_content_length"]
        }
    except Exception as e:
        logger.error(f"Error indexing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-index")
async def batch_index_content(content_data: BatchContentIndexRequest):
    """
    Index multiple pieces of content at once
    """
    try:
        logger.info(f"Batch indexing {len(content_data.contents)} content items")

        # Convert to the format expected by the service
        contents = [
            {
                "content": item.content,
                "chapter_id": item.chapter_id,
                "section_title": item.section_title,
                "source_url": item.source_url
            }
            for item in content_data.contents
        ]

        results = await rag_service.batch_index_content(contents)

        return {
            "success": True,
            "indexed_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error batch indexing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def content_status():
    """
    Get status of the content indexing system
    """
    try:
        # Test Qdrant connection and get point count
        point_count = await qdrant_service.count_points()

        return {
            "status": "ready",
            "indexed_documents_count": point_count,
            "services": {
                "qdrant": await qdrant_service.test_connection(),
                "mistral": await mistral_service.test_connection()
            }
        }
    except Exception as e:
        logger.error(f"Error getting content status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest-from-docs")
async def ingest_from_docs(background_tasks: BackgroundTasks):
    """
    Ingest content from the docs directory (for textbook content)
    This would read the Docusaurus docs and index them
    """
    try:
        # This would typically read from the docs directory
        # and index all the content for RAG
        docs_path = os.getenv("DOCS_PATH", "/app/docs")  # Default path in Docker

        if not os.path.exists(docs_path):
            raise HTTPException(status_code=404, detail=f"Docs path not found: {docs_path}")

        # This is a simplified version - in a real implementation, you'd want to
        # parse all the markdown files and extract content properly
        import os
        import glob

        markdown_files = glob.glob(f"{docs_path}/**/*.md", recursive=True)
        markdown_files += glob.glob(f"{docs_path}/**/*.mdx", recursive=True)

        content_items = []
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract chapter/section info from file path
                relative_path = os.path.relpath(file_path, docs_path)
                chapter_id = relative_path.replace('/', '_').replace('\\', '_').replace('.md', '').replace('.mdx', '')
                section_title = os.path.basename(file_path).replace('.md', '').replace('.mdx', '')

                content_items.append({
                    "content": content,
                    "chapter_id": chapter_id,
                    "section_title": section_title,
                    "source_url": f"/docs/{relative_path}"
                })
            except Exception as e:
                logger.warning(f"Could not process file {file_path}: {str(e)}")

        # Index all content items
        results = await rag_service.batch_index_content(content_items)

        return {
            "success": True,
            "processed_files": len(content_items),
            "indexed_documents": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error ingesting from docs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/personalize")
async def personalize_content(personalize_data: PersonalizeContentRequest):
    """
    Get personalized content based on user background
    """
    try:
        # In a real implementation, this would adapt content based on user background
        # For now, we'll return the original content with some basic adaptation info
        logger.info(f"Personalizing content for chapter {personalize_data.chapter_id} for user {personalize_data.user_id}")

        # This would typically search for content related to the chapter_id
        # and adapt it based on the user's background
        # For now, we'll return a placeholder response
        return {
            "chapter_id": personalize_data.chapter_id,
            "user_id": personalize_data.user_id,
            "adaptation_info": "Content adaptation based on user background would happen here",
            "original_content": "Original textbook content would be returned here, adapted based on user background",
            "user_background": personalize_data.user_background
        }
    except Exception as e:
        logger.error(f"Error personalizing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))