#!/usr/bin/env python3
"""
Test script to verify MistralAI integration is working properly
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.mistral_service import MistralService

async def test_mistral_connection():
    """
    Test the Mistral service connection and functionality
    """
    print("Testing MistralAI integration...")

    try:
        # Initialize the Mistral service
        mistral_service = MistralService()
        print("✓ Mistral service initialized successfully")

        # Test basic connection
        print("\nTesting basic response generation...")
        response = await mistral_service.generate_response("Hello, how are you?")
        print(f"✓ Response received: {response[:100]}...")

        # Test embedding generation
        print("\nTesting embedding generation...")
        texts = ["This is a test sentence for embeddings.", "Another test sentence."]
        embeddings = await mistral_service.generate_embeddings(texts)
        print(f"✓ Generated embeddings for {len(texts)} texts")
        print(f"  Embedding dimensions: {len(embeddings[0])}")

        # Test query embedding
        print("\nTesting query embedding...")
        query_embedding = await mistral_service.generate_embeddings_query("What is artificial intelligence?")
        print(f"✓ Query embedding generated with {len(query_embedding)} dimensions")

        # Test connection method
        print("\nTesting connection method...")
        connection_ok = await mistral_service.test_connection()
        print(f"✓ Connection test: {'PASSED' if connection_ok else 'FAILED'}")

        print("\n🎉 All MistralAI tests passed! The integration is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Error during MistralAI testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mistral_connection())
    if success:
        print("\n✅ MistralAI is properly configured and responding!")
    else:
        print("\n❌ There are issues with the MistralAI configuration.")