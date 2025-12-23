import logging
from typing import List
from fastembed import TextEmbedding
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize local embedding service using fastembed
        """
        self.model_name = model_name
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the local embedding model"""
        try:
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Local embedding model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing local embedding model: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using local model
        """
        try:
            # Convert generator to list to ensure all embeddings are generated
            embeddings = list(self._model.embed(texts))
            return [embedding.tolist() if hasattr(embedding, 'tolist') else embedding for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise

    async def generate_embeddings_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query using local model
        """
        try:
            embeddings = list(self._model.embed([text]))
            embedding = embeddings[0]
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"Error generating local query embedding: {e}")
            raise

    async def test_connection(self) -> bool:
        """
        Test local embedding generation
        """
        try:
            test_embedding = await self.generate_embeddings_query("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Local embedding connection test failed: {e}")
            return False