from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        # Get Qdrant configuration
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

        # Initialize Qdrant client
        if self.api_key:
            self.client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        else:
            self.client = AsyncQdrantClient(url=self.url)

        # Collection vector size - check if using OpenRouter (1536) or Cohere (1024)
        # OpenAI text-embedding-ada-002 produces 1536-dimensional vectors
        embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "")
        if "ada-002" in embedding_model.lower() or os.getenv("USE_OPENROUTER_EMBEDDINGS", "false").lower() == "true":
            self.vector_size = 1536  # OpenAI text-embedding-ada-002 dimensions
        else:
            self.vector_size = 1024  # Cohere multilingual embeddings are 1024 dimensions (fallback)

    async def create_collection(self):
        """
        Create the collection if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    async def store_embedding(self,
                            vector: List[float],
                            content: str,
                            chapter_id: str,
                            section_title: str,
                            source_url: str,
                            additional_metadata: Dict[str, Any] = None) -> str:
        """
        Store an embedding with its metadata in Qdrant
        """
        try:
            # Create a unique ID for the document
            doc_id = str(uuid.uuid4())

            from datetime import datetime
            payload = {
                "content": content,
                "chapter_id": chapter_id,
                "section_title": section_title,
                "source_url": source_url,
                "created_at": datetime.utcnow().isoformat()
            }

            # Merge additional metadata if provided
            if additional_metadata:
                payload.update(additional_metadata)

            # Store the embedding
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )

            logger.info(f"Stored embedding for chapter {chapter_id}, section {section_title}")
            return doc_id
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise

    async def search_similar(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in Qdrant
        """
        try:
            # Search for similar vectors using search method (compatible with qdrant-client 1.7.0)
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "content": result.payload["content"],
                    "chapter_id": result.payload["chapter_id"],
                    "section_title": result.payload["section_title"],
                    "source_url": result.payload["source_url"],
                    "score": result.score
                })

            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            raise

    async def delete_collection(self):
        """
        Delete the collection (useful for testing/refreshing)
        """
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    async def clear_collection(self):
        """
        Clear all points from the collection without deleting the collection itself
        """
        try:
            # Delete and recreate collection to clear all points
            await self.delete_collection()
            await self.create_collection()
            logger.info(f"Cleared all points from collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    async def count_points(self) -> int:
        """
        Get the number of points (documents) in the collection
        """
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error counting points: {str(e)}")
            raise

    async def test_connection(self) -> bool:
        """
        Test connection to Qdrant
        """
        try:
            # Try to get collections list to test connection
            collections = await self.client.get_collections()
            logger.info(f"Qdrant connection successful. Found {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {str(e)}")
            return False