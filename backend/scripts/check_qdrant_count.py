#!/usr/bin/env python3
"""
Simple script to check current Qdrant point count
"""
import asyncio
import httpx
import os

async def check_point_count():
    """Check the current point count in Qdrant"""
    try:
        # Use the same configuration as in the ingestion script
        url = os.getenv("QDRANT_URL", "https://b7fd2e52-546c-4747-8f11-3c0eecabe4f6.us-east4-0.gcp.cloud.qdrant.io:6333")
        api_key = os.getenv("QDRANT_API_KEY")  # This should be set in environment
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

        headers = {}
        if api_key:
            headers["api-key"] = api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/collections/{collection_name}",
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                point_count = data["result"]["points_count"]
                print(f"Current points in Qdrant: {point_count}")
                return point_count
            else:
                print(f"Could not fetch point count: {response.status_code}")
                print(f"Response: {response.text}")
                return None
    except Exception as e:
        print(f"Error checking point count: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(check_point_count())