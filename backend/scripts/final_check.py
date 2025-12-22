#!/usr/bin/env python3
"""
Script to check final Qdrant point count after ingestion
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.qdrant_service import QdrantService

async def main():
    print("Checking final point count in Qdrant...")

    qdrant_service = QdrantService()

    try:
        # Try direct collection info to bypass validation issues
        collection_info = await qdrant_service.client.get_collection(qdrant_service.collection_name)
        print(f"Collection points count: {collection_info.points_count}")
        print(f"Collection name: {qdrant_service.collection_name}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
    except Exception as e:
        print(f"Could not get collection info directly: {e}")

        # Try the count method again
        try:
            count = await qdrant_service.count_points()
            print(f"Points in collection (via count_points): {count}")
        except Exception as e2:
            print(f"Could not count points: {e2}")

if __name__ == "__main__":
    asyncio.run(main())