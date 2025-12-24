from typing import List, Dict, Any
import logging
from .openrouter_service import OpenRouterService
from .qdrant_service import QdrantService
from .local_embedding_service import LocalEmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGService:
    def __init__(self, openrouter_service: OpenRouterService, qdrant_service: QdrantService):
        self.openrouter_service = openrouter_service
        self.qdrant_service = qdrant_service
        try:
            self.local_embedding_service = LocalEmbeddingService()
            self.local_embeddings_available = True
        except Exception as e:
            logger.warning(f"Local embeddings not available: {e}")
            self.local_embedding_service = None
            self.local_embeddings_available = False
        self.use_local_embeddings = False  # Will switch to local if OpenRouter fails

    def _switch_to_local_embeddings(self):
        """Switch to local embeddings if OpenRouter is failing and local embeddings are available"""
        if not self.use_local_embeddings and self.local_embeddings_available:
            logger.warning("Switching to local embeddings due to API issues")
            self.use_local_embeddings = True
        elif not self.local_embeddings_available:
            logger.error("Cannot switch to local embeddings - not available in this environment")

    async def _generate_embeddings(self, texts: List[str]):
        """Generate embeddings with fallback to local embeddings"""
        if self.use_local_embeddings and self.local_embeddings_available:
            return await self.local_embedding_service.generate_embeddings(texts)
        else:
            try:
                return await self.openrouter_service.generate_embeddings(texts)
            except Exception as e:
                if ("rate limit" in str(e).lower() or "429" in str(e)) and self.local_embeddings_available:
                    self._switch_to_local_embeddings()
                    logger.info("Retrying with local embeddings...")
                    return await self.local_embedding_service.generate_embeddings(texts)
                else:
                    raise

    async def _generate_query_embedding(self, text: str):
        """Generate query embedding with fallback to local embeddings"""
        if self.use_local_embeddings and self.local_embeddings_available:
            return await self.local_embedding_service.generate_embeddings_query(text)
        else:
            try:
                return await self.openrouter_service.generate_embeddings_query(text)
            except Exception as e:
                if ("rate limit" in str(e).lower() or "429" in str(e)) and self.local_embeddings_available:
                    self._switch_to_local_embeddings()
                    logger.info("Retrying with local embeddings...")
                    return await self.local_embedding_service.generate_embeddings_query(text)
                else:
                    raise

    async def index_content(self, content: str, chapter_id: str, section_title: str, source_url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Index content by creating embeddings and storing in Qdrant
        """
        try:
            # Create embeddings for the content
            embeddings = await self._generate_embeddings([content])

            if not embeddings or len(embeddings) == 0:
                raise ValueError("Failed to generate embeddings for content")

            # Store in Qdrant
            document_id = await self.qdrant_service.store_embedding(
                vector=embeddings[0],
                content=content,
                chapter_id=chapter_id,
                section_title=section_title,
                source_url=source_url,
                additional_metadata=metadata
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
            query_embedding = await self._generate_query_embedding(query)

            # Search for similar content in Qdrant
            similar_docs = await self.qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=5  # Get top 5 similar documents
            )

            if not similar_docs:
                # If no similar documents found, return a response indicating this
                # Use OpenRouter for generation (this is less likely to hit rate limits since it's one call)
                try:
                    answer = await self.openrouter_service.generate_response(query)
                except Exception as e:
                    answer = f"Could not generate response: {str(e)}"
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
            try:
                answer = await self.openrouter_service.generate_response(query, context)
            except Exception as e:
                answer = f"Could not generate response: {str(e)}"

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