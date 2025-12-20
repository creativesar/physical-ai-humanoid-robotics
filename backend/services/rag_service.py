from typing import List, Dict, Any
import logging
from .mistral_service import MistralService
from .qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, mistral_service: MistralService, qdrant_service: QdrantService):
        self.mistral_service = mistral_service
        self.qdrant_service = qdrant_service

    async def index_content(self, content: str, chapter_id: str, section_title: str, source_url: str) -> Dict[str, Any]:
        """
        Index content by creating embeddings and storing in Qdrant
        """
        try:
            # Create embeddings for the content
            embeddings = await self.mistral_service.generate_embeddings([content])

            if not embeddings or len(embeddings) == 0:
                raise ValueError("Failed to generate embeddings for content")

            # Store in Qdrant
            document_id = await self.qdrant_service.store_embedding(
                vector=embeddings[0],
                content=content,
                chapter_id=chapter_id,
                section_title=section_title,
                source_url=source_url
            )

            return {
                "document_id": document_id,
                "indexed_content_length": len(content)
            }
        except Exception as e:
            logger.error(f"Error indexing content: {str(e)}")
            raise

    async def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a query using RAG: retrieve relevant context and generate response
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.mistral_service.generate_embeddings_query(query)

            # Search for similar content in Qdrant
            similar_docs = await self.qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=5  # Get top 5 similar documents
            )

            if not similar_docs:
                # If no similar documents found, return a response indicating this
                answer = await self.mistral_service.generate_response(query)
                return {
                    "answer": answer,
                    "sources": [],
                    "query_embedding": query_embedding
                }

            # Combine the content from similar documents as context
            context_parts = []
            sources = []

            for doc in similar_docs:
                context_parts.append(doc["content"])
                sources.append({
                    "title": doc["section_title"],
                    "url": doc["source_url"],
                    "chapter_id": doc["chapter_id"],
                    "relevance_score": doc["score"]
                })

            # Combine all context parts
            context = "\n\n".join(context_parts)

            # Generate response using the context
            answer = await self.mistral_service.generate_response(query, context)

            return {
                "answer": answer,
                "sources": sources,
                "query_embedding": query_embedding
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def batch_index_content(self, contents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Index multiple content items at once
        """
        results = []
        for content_data in contents:
            result = await self.index_content(
                content=content_data["content"],
                chapter_id=content_data["chapter_id"],
                section_title=content_data["section_title"],
                source_url=content_data["source_url"]
            )
            results.append(result)
        return results

    async def initialize_collection(self):
        """
        Initialize the Qdrant collection if needed
        """
        await self.qdrant_service.create_collection()