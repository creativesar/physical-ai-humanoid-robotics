#!/usr/bin/env python3
"""
Content Ingestion Script for Physical AI & Humanoid Robotics Textbook

This script ingests the textbook content from the docs directory
and indexes it in Qdrant for RAG functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utils.content_ingestor import ContentIngestor
from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

async def main():
    """
    Main function to run the content ingestion process
    """
    print("Starting content ingestion process for Physical AI & Humanoid Robotics Textbook...")

    # Initialize services
    try:
        mistral_service = MistralService()
        qdrant_service = QdrantService()
        rag_service = RAGService(mistral_service, qdrant_service)

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

    # Confirm with user before proceeding
    response = input(f"\nReady to ingest {len(markdown_files)} files into Qdrant. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Ingestion cancelled by user.")
        return

    # Ingest content
    print("\nStarting ingestion process...")
    results = await ingestor.ingest_from_directory(docs_path)

    print(f"\n[SUCCESS] Ingestion completed!")
    print(f"  - Successfully processed: {results['total_processed']} files")
    print(f"  - Failed: {results['total_failed']} files")

    if results['failed']:
        print(f"\nFailed files:")
        for failed in results['failed']:
            print(f"  - {failed['file']}: {failed['error']}")

    # Show final count
    try:
        point_count = await qdrant_service.count_points()
        print(f"\n[SUCCESS] Total documents now in Qdrant: {point_count}")
    except Exception as e:
        print(f"[ERROR] Could not get final count from Qdrant: {e}")

if __name__ == "__main__":
    asyncio.run(main())