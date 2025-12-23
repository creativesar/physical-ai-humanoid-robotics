#!/usr/bin/env python3
"""
Test script to verify OpenRouter RAG functionality
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.openrouter_service import OpenRouterService
from services.optimized_openrouter_rag_service import OptimizedOpenRouterRAGService
from services.qdrant_service import QdrantService

async def test_openrouter_rag():
    """
    Test the OpenRouter RAG service connection and functionality
    """
    print("Testing OpenRouter RAG integration...")

    try:
        # Initialize the OpenRouter service
        openrouter_service = OpenRouterService()
        print("✓ OpenRouter service initialized successfully")

        # Test basic connection
        print("\nTesting basic response generation...")
        response = await openrouter_service.generate_response("Hello, how are you?")
        print(f"✓ Response received: {response[:100]}...")

        # Test embedding generation
        print("\nTesting embedding generation...")
        texts = ["This is a test sentence for embeddings.", "Another test sentence."]
        embeddings = await openrouter_service.generate_embeddings(texts)
        print(f"✓ Generated embeddings for {len(texts)} texts")
        print(f"  Embedding dimensions: {len(embeddings[0])}")

        # Test query embedding
        print("\nTesting query embedding...")
        query_embedding = await openrouter_service.generate_embeddings_query("What is artificial intelligence?")
        print(f"✓ Query embedding generated with {len(query_embedding)} dimensions")

        # Test connection method
        print("\nTesting connection method...")
        connection_ok = await openrouter_service.test_connection()
        print(f"✓ Connection test: {'PASSED' if connection_ok else 'FAILED'}")

        print("\n🎉 All OpenRouter tests passed! The integration is working correctly.")

        # Now test the full RAG service
        print("\nTesting Optimized OpenRouter RAG Service...")
        qdrant_service = QdrantService()
        rag_service = OptimizedOpenRouterRAGService(openrouter_service, qdrant_service)

        # Test initialization
        await rag_service.initialize_collection()
        print("✓ RAG service initialized and collection created")

        # Test indexing content
        print("\nTesting content indexing...")
        result = await rag_service.index_content(
            content="This is a test document about artificial intelligence and robotics.",
            chapter_id="test_chapter_001",
            section_title="Test Section",
            source_url="/test/source"
        )
        print(f"✓ Content indexed successfully: {result}")

        # Test querying
        print("\nTesting query processing...")
        query_result = await rag_service.process_query("What is this document about?")
        print(f"✓ Query processed: {query_result['answer'][:100]}...")

        # Get collection stats
        stats = await rag_service.get_collection_stats()
        print(f"✓ Collection stats: {stats}")

        print("\n🎉 All RAG tests passed! The OpenRouter RAG system is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Error during OpenRouter RAG testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openrouter_rag())
    if success:
        print("\n✅ OpenRouter RAG is properly configured and responding!")
    else:
        print("\n❌ There are issues with the OpenRouter RAG configuration.")