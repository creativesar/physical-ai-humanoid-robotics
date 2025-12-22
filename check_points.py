#!/usr/bin/env python3
"""
Simple script to check current points in Qdrant
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.qdrant_service import QdrantService

async def main():
    print("Checking current points in Qdrant...")

    qdrant_service = QdrantService()

    try:
        count = await qdrant_service.count_points()
        print(f"Current number of points in Qdrant: {count}")
    except Exception as e:
        print(f"Error counting points: {e}")
        # Let's try to get collection info directly
        try:
            collection_info = await qdrant_service.client.get_collection(qdrant_service.collection_name)
            print(f"Collection points count: {collection_info.points_count}")
        except Exception as e2:
            print(f"Could not get collection info: {e2}")

    # Let's also test connection
    try:
        connection_ok = await qdrant_service.test_connection()
        print(f"Qdrant connection: {'OK' if connection_ok else 'FAILED'}")
    except Exception as e:
        print(f"Connection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())