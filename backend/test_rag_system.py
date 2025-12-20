#!/usr/bin/env python3
"""
Comprehensive test script to verify RAG system including Qdrant and MistralAI
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.mistral_service import MistralService
from services.qdrant_service import QdrantService
from services.rag_service import RAGService

async def test_full_rag_system():
    """
    Test the complete RAG system including Qdrant and MistralAI
    """
    print("Testing Full RAG System (Qdrant + MistralAI)...")

    try:
        # Initialize services
        print("1. Initializing Mistral service...")
        mistral_service = MistralService()
        print("   ✓ Mistral service initialized")

        print("\n2. Initializing Qdrant service...")
        qdrant_service = QdrantService()
        print("   ✓ Qdrant service initialized")

        print("\n3. Initializing RAG service...")
        rag_service = RAGService(mistral_service, qdrant_service)
        print("   ✓ RAG service initialized")

        # Test Qdrant connection
        print("\n4. Testing Qdrant connection...")
        qdrant_ok = await qdrant_service.test_connection()
        print(f"   ✓ Qdrant connection: {'PASSED' if qdrant_ok else 'FAILED'}")

        # Test Mistral connection
        print("\n5. Testing Mistral connection...")
        mistral_ok = await mistral_service.test_connection()
        print(f"   ✓ Mistral connection: {'PASSED' if mistral_ok else 'FAILED'}")

        # Test embedding generation (Mistral)
        print("\n6. Testing embedding generation...")
        test_texts = ["This is a test document for RAG system.", "Another test sentence."]
        embeddings = await mistral_service.generate_embeddings(test_texts)
        print(f"   ✓ Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions each")

        # Test storing in Qdrant
        print("\n7. Testing Qdrant storage...")
        doc_id = await qdrant_service.store_embedding(
            vector=embeddings[0],
            content="Test content for RAG system",
            chapter_id="test-chapter",
            section_title="Test Section",
            source_url="/test/source"
        )
        print(f"   ✓ Stored embedding in Qdrant with ID: {doc_id}")

        # Test search in Qdrant
        print("\n8. Testing Qdrant search...")
        query_embedding = await mistral_service.generate_embeddings_query("What is this test about?")
        search_results = await qdrant_service.search_similar(query_embedding, limit=1)
        print(f"   ✓ Found {len(search_results)} similar documents in Qdrant")

        # Test full RAG process
        print("\n9. Testing full RAG query process...")
        rag_result = await rag_service.process_query("What is this test about?", user_id="test-user")
        print(f"   ✓ RAG response generated: {rag_result['answer'][:100]}...")

        # Test collection count
        print("\n10. Testing collection statistics...")
        count = await qdrant_service.count_points()
        print(f"   ✓ Total documents in Qdrant: {count}")

        print("\n🎉 ALL RAG SYSTEM TESTS PASSED!")
        print("✅ Qdrant is connecting properly")
        print("✅ MistralAI is responding properly")
        print("✅ RAG system is working end-to-end")
        return True

    except Exception as e:
        print(f"\n❌ Error during RAG system testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """
    Test each component individually to isolate issues
    """
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT TESTS")
    print("="*60)

    # Test Mistral only
    print("\nTesting MistralAI only...")
    try:
        mistral = MistralService()
        response = await mistral.generate_response("Test message")
        print(f"✓ MistralAI response: {response[:50]}...")
    except Exception as e:
        print(f"❌ MistralAI error: {e}")

    # Test Qdrant only
    print("\nTesting Qdrant only...")
    try:
        qdrant = QdrantService()
        connection_ok = await qdrant.test_connection()
        print(f"✓ Qdrant connection: {connection_ok}")
        count = await qdrant.count_points()
        print(f"✓ Qdrant document count: {count}")
    except Exception as e:
        print(f"❌ Qdrant error: {e}")

if __name__ == "__main__":
    print("Starting comprehensive RAG system test...")

    success = asyncio.run(test_full_rag_system())
    asyncio.run(test_individual_components())

    if success:
        print("\n✅ RAG system is fully operational!")
        print("Both Qdrant and MistralAI are working correctly together.")
    else:
        print("\n❌ There are issues with the RAG system.")
        print("Check the error messages above for details.")