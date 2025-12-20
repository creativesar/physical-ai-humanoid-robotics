import mistralai
from mistralai import Mistral
from typing import List, Dict, Any
import logging
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralService:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        self.client = Mistral(api_key=api_key)
        self.embedding_model = "mistral-embed"  # Mistral's embedding model
        self.generation_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")  # Default model

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using Mistral
        """
        try:
            response = await self.client.embeddings.create_async(
                model=self.embedding_model,
                inputs=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Mistral embedding error for {len(texts)} texts: {str(e)}")
            if hasattr(e, 'http_status'):
                logger.error(f"HTTP Status: {e.http_status}")
                if e.http_status == 401:
                    logger.error("Authentication failed - check your MISTRAL_API_KEY in the .env file")
            raise

    async def generate_embeddings_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query using Mistral
        """
        try:
            response = await self.client.embeddings.create_async(
                model=self.embedding_model,
                inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Mistral query embedding error: {str(e)}")
            if hasattr(e, 'http_status'):
                logger.error(f"HTTP Status: {e.http_status}")
                if e.http_status == 401:
                    logger.error("Authentication failed - check your MISTRAL_API_KEY in the .env file")
            raise

    async def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate a response using Mistral's language model
        """
        try:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nProvide a very concise answer in 1-2 sentences. If context is insufficient, say so briefly."

            chat_response = await self.client.chat.complete_async(
                model=self.generation_model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                temperature=0.1,  # Very low temperature for focused responses
                max_tokens=100    # Further limit tokens for brevity
            )

            return chat_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral chat response error: {str(e)}")
            if hasattr(e, 'http_status'):
                logger.error(f"HTTP Status: {e.http_status}")
                if e.http_status == 401:
                    logger.error("Authentication failed - check your MISTRAL_API_KEY in the .env file")
            if "rate limit" in str(e).lower():
                logger.warning("Mistral rate limit hit")
            raise

    async def test_connection(self) -> bool:
        """
        Test connection to Mistral API
        """
        try:
            # Test with a simple response
            test_response = await self.generate_response("Test connection")
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"Mistral connection test failed: {str(e)}")
            if hasattr(e, 'http_status'):
                logger.error(f"HTTP Status: {e.http_status}")
                if e.http_status == 401:
                    logger.error("Authentication failed - check your MISTRAL_API_KEY in the .env file")
            return False