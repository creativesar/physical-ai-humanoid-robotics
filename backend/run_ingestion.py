#!/usr/bin/env python3
"""
Direct content ingestion script without user confirmation
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utils.content_ingestor import ContentIngestor
from services.optimized_openrouter_rag_service import OptimizedOpenRouterRAGService
from services.qdrant_service import QdrantService
from services.openrouter_service import OpenRouterService

async def main():
    """
    Main function to run the content ingestion process without user confirmation
    """
    print("Starting content ingestion process for Physical AI & Humanoid Robotics Textbook...")

    # Initialize services
    try:
        openrouter_service = OpenRouterService()
        qdrant_service = QdrantService()
        rag_service = OptimizedOpenRouterRAGService(openrouter_service, qdrant_service)

        print("[SUCCESS] Services initialized successfully")
    except Exception as e:
        print(f"[ERROR] Error initializing services: {e}")
        return

    # Initialize content ingestor
    ingestor = ContentIngestor(rag_service)

    # Initialize the Qdrant collection
    try:
        await rag_service.initialize_collection()
        print("[SUCCESS] Qdrant collection initialized")
    except Exception as e:
        print(f"[ERROR] Error initializing Qdrant collection: {e}")
        return

    # Get docs path from environment or use default
    docs_path = os.getenv("DOCS_PATH", "../frontend/docs")

    if not os.path.exists(docs_path):
        print(f"[ERROR] Docs path does not exist: {docs_path}")
        print("Please make sure the docs directory exists and contains the textbook content.")
        return

    print(f"[SUCCESS] Found docs directory: {docs_path}")

    # Count files to be processed
    import glob
    markdown_files = []
    markdown_files.extend(glob.glob(f"{docs_path}/**/*.md", recursive=True))
    markdown_files.extend(glob.glob(f"{docs_path}/**/*.mdx", recursive=True))

    print(f"Found {len(markdown_files)} markdown files to process")

    if len(markdown_files) == 0:
        print("No markdown files found in the docs directory.")
        return

    # Check current point count
    try:
        current_count = await qdrant_service.count_points()
        print(f"Current points in Qdrant: {current_count}")
    except Exception as e:
        current_count = 0
        print(f"Could not get current point count: {e}")

    print(f"\nAutomatically proceeding with ingestion of {len(markdown_files)} files into Qdrant...")
    print("[WARNING] This will CLEAR all existing data in the collection!")

    # Clear existing collection to avoid duplicates
    print("\n[INFO] Clearing existing collection...")
    try:
        await qdrant_service.clear_collection()
        print("[SUCCESS] Collection cleared successfully")
    except Exception as e:
        print(f"[ERROR] Failed to clear collection: {e}")
        return

    # Ingest content
    print("\n[INFO] Starting ingestion process...")
    results = await ingestor.ingest_from_directory(docs_path)

    print(f"\n[SUCCESS] Ingestion completed!")
    print(f"  - Successfully processed: {results['total_files_processed']} files")
    print(f"  - Total chunks created: {results['total_chunks_created']}")
    print(f"  - Failed: {results['total_failed']} files")
    if results['total_files_processed'] > 0:
        avg_chunks = results['total_chunks_created'] / results['total_files_processed']
        print(f"  - Average chunks per file: {avg_chunks:.1f}")

    if results['failed']:
        print(f"\nFailed files:")
        for failed in results['failed']:
            print(f"  - {failed['file']}: {failed['error']}")

    # Show final count
    try:
        point_count = await qdrant_service.count_points()
        print(f"\n[SUCCESS] Total points now in Qdrant: {point_count}")
    except Exception as e:
        print(f"[ERROR] Could not get final count from Qdrant: {e}")

if __name__ == "__main__":
    asyncio.run(main())