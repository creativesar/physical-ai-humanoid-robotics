from typing import List, Dict, Any
import logging
import asyncio
from .openrouter_service import OpenRouterService
from .qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOpenRouterRAGService:
    def __init__(self, openrouter_service: OpenRouterService, qdrant_service: QdrantService):
        self.openrouter_service = openrouter_service
        self.qdrant_service = qdrant_service
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        # Batch size for processing multiple items at once
        self.batch_size = 10

    async def index_content(self, content: str, chapter_id: str, section_title: str, source_url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Index content by creating embeddings and storing in Qdrant using OpenRouter with optimizations
        """
        try:
            # Create embeddings for the content using OpenRouter
            embeddings = await self.openrouter_service.generate_embeddings([content])

            if not embeddings or len(embeddings) == 0:
                raise ValueError("Failed to generate embeddings for content using OpenRouter")

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
            logger.error(f"Error indexing content with OpenRouter: {str(e)}")
            raise

    async def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a query using optimized RAG: retrieve relevant context and generate response
        """
        try:
            # Generate embedding for the query using OpenRouter
            query_embedding = await self.openrouter_service.generate_embeddings_query(query)

            # Search for similar content in Qdrant with increased limit for better results
            similar_docs = await self.qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=10  # Increased from 5 to 10 for better context
            )

            if not similar_docs:
                # If no similar documents found, return a response indicating this
                answer = await self.openrouter_service.generate_response(query)
                return {
                    "answer": answer,
                    "sources": [],
                    "query_embedding": query_embedding
                }

            # Combine the content from similar documents as context
            # Sort by relevance score to prioritize better matches
            sorted_docs = sorted(similar_docs, key=lambda x: x["score"], reverse=True)

            # Limit context to top 7 most relevant docs to provide more comprehensive context
            top_docs = sorted_docs[:7]

            context_parts = []
            sources = []

            for doc in top_docs:
                context_parts.append(doc["content"])
                sources.append({
                    "title": doc["section_title"],
                    "url": doc["source_url"],
                    "chapter_id": doc["chapter_id"],
                    "relevance_score": doc["score"]
                })

            # Combine all context parts
            context = "\n\n".join(context_parts)

            # Generate response using the context with OpenRouter
            answer = await self.openrouter_service.generate_response(query, context)

            return {
                "answer": answer,
                "sources": sources,
                "query_embedding": query_embedding
            }
        except Exception as e:
            logger.error(f"Error processing query with OpenRouter: {str(e)}")
            raise

    async def batch_index_content(self, contents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Index multiple content items efficiently in batches
        """
        results = []

        # Process in batches to optimize performance
        for i in range(0, len(contents), self.batch_size):
            batch = contents[i:i + self.batch_size]

            # Process batch concurrently
            batch_tasks = []
            for content_data in batch:
                task = self.index_content(
                    content=content_data["content"],
                    chapter_id=content_data["chapter_id"],
                    section_title=content_data["section_title"],
                    source_url=content_data["source_url"]
                )
                batch_tasks.append(task)

            # Execute batch tasks concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results and handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch indexing error: {str(result)}")
                    results.append({"error": str(result)})
                else:
                    results.append(result)

        return results

    async def initialize_collection(self):
        """
        Initialize the Qdrant collection if needed
        """
        await self.qdrant_service.create_collection()

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection for performance monitoring
        """
        try:
            count = await self.qdrant_service.count_points()
            return {
                "total_documents": count,
                "service": "OpenRouter",
                "model": self.openrouter_service.generation_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}