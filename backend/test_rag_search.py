#!/usr/bin/env python3
"""
Test script to verify RAG search functionality
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

async def test_rag_search():
    """
    Test RAG search functionality with specific queries
    """
    print("Testing RAG search functionality...")

    # Initialize services
    try:
        mistral_service = MistralService()
        qdrant_service = QdrantService()
        rag_service = RAGService(mistral_service, qdrant_service)

        print("[SUCCESS] Services initialized successfully")
    except Exception as e:
        print(f"[ERROR] Error initializing services: {e}")
        return

    # Test queries that should match specific content in your textbook
    test_queries = [
        "ROS 2 architecture",
        "Humanoid robotics",
        "Gazebo simulation",
        "NVIDIA Isaac",
        "Vision Language Action",
        "module 1",
        "module 2",
        "module 3",
        "module 4"
    ]

    print(f"\nTesting {len(test_queries)} search queries...")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            # Process the query through RAG
            result = await rag_service.process_query(query)

            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Number of sources found: {len(result['sources'])}")

            if result['sources']:
                print("   Top source:")
                top_source = result['sources'][0]
                print(f"     Title: {top_source['title']}")
                print(f"     Chapter ID: {top_source['chapter_id']}")
                print(f"     Relevance Score: {top_source['relevance_score']:.4f}")
                print(f"     Content preview: {top_source['content'][:100]}...")
            else:
                print("   No relevant sources found!")

        except Exception as e:
            print(f"   [ERROR] Search failed: {e}")

    # Show total points in collection
    try:
        point_count = await qdrant_service.count_points()
        print(f"\n[SUCCESS] Total documents in Qdrant: {point_count}")
    except Exception as e:
        print(f"[ERROR] Could not get point count: {e}")

if __name__ == "__main__":
    asyncio.run(test_rag_search())