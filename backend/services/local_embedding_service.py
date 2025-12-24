import logging
from typing import List, Optional
import asyncio
import time

# Optional import - fastembed may not be installed in serverless environments
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize local embedding service using fastembed
        """
        if not FASTEMBED_AVAILABLE:
            logger.warning("fastembed not available - local embeddings disabled")
            self._model = None
            self.model_name = None
            return

        self.model_name = model_name
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the local embedding model"""
        if not FASTEMBED_AVAILABLE:
            return

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
        if not FASTEMBED_AVAILABLE or self._model is None:
            raise RuntimeError("Local embeddings not available - fastembed not installed")

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
        if not FASTEMBED_AVAILABLE or self._model is None:
            raise RuntimeError("Local embeddings not available - fastembed not installed")

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
        if not FASTEMBED_AVAILABLE or self._model is None:
            logger.warning("Local embeddings not available for testing")
            return False

        try:
            test_embedding = await self.generate_embeddings_query("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Local embedding connection test failed: {e}")
            return False