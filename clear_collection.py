#!/usr/bin/env python3
"""
Simple script to clear Qdrant collection and check points
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.qdrant_service import QdrantService

async def main():
    print("Connecting to Qdrant...")

    qdrant_service = QdrantService()

    try:
        # Try to delete and recreate collection
        print("Attempting to recreate collection...")
        await qdrant_service.delete_collection()
        print("Old collection deleted.")
    except Exception as e:
        print(f"Collection might not exist yet, creating new one: {e}")

    try:
        await qdrant_service.create_collection()
        print("New collection created successfully.")

        # Check the count
        try:
            count = await qdrant_service.count_points()
            print(f"Points in new collection: {count}")
        except Exception as e:
            print(f"Could not count points: {e}")

    except Exception as e:
        print(f"Error creating collection: {e}")

if __name__ == "__main__":
    asyncio.run(main())