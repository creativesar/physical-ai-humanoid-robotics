from openai import AsyncOpenAI
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

class OpenRouterService:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        # Configure OpenAI to use OpenRouter
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Use a model that works with OpenRouter (e.g., OpenAI's text-embedding-ada-002 for embeddings)
        self.embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.generation_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # Default model for generation

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using OpenRouter
        """
        try:
            # For batch processing, we'll process each text individually to avoid token limits
            embeddings = []
            for text in texts:
                # Truncate text if it's too long (OpenAI has token limits)
                truncated_text = text[:8192]  # OpenAI models have token limits
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=truncated_text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            logger.error(f"OpenRouter embedding error for {len(texts)} texts: {str(e)}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status: {e.status_code}")
                if e.status_code == 401:
                    logger.error("Authentication failed - check your OPENROUTER_API_KEY in the .env file")
            raise

    async def generate_embeddings_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query using OpenRouter
        """
        try:
            # Truncate text if it's too long
            truncated_text = text[:8192]  # OpenAI models have token limits
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenRouter query embedding error: {str(e)}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status: {e.status_code}")
                if e.status_code == 401:
                    logger.error("Authentication failed - check your OPENROUTER_API_KEY in the .env file")
            raise

    async def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate a response using OpenRouter's language model
        """
        try:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a comprehensive, detailed response with specific information relevant to the query. Structure your response in a formal, academic manner with clear explanations. If the context is insufficient, acknowledge this and provide general information based on your knowledge while indicating the limitations of the provided context."

            response = await self.client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                temperature=0.35,  # Moderate temperature for balanced, detailed responses
                max_tokens=500    # Increased token limit for more comprehensive answers
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter chat response error: {str(e)}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status: {e.status_code}")
                if e.status_code == 401:
                    logger.error("Authentication failed - check your OPENROUTER_API_KEY in the .env file")
            if "rate limit" in str(e).lower():
                logger.warning("OpenRouter rate limit hit")
            raise

    async def test_connection(self) -> bool:
        """
        Test connection to OpenRouter API
        """
        try:
            # Test with a simple response
            test_response = await self.generate_response("Test connection")
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {str(e)}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status: {e.status_code}")
                if e.status_code == 401:
                    logger.error("Authentication failed - check your OPENROUTER_API_KEY in the .env file")
            return False
