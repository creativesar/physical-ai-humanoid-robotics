#!/usr/bin/env python3
"""
Simple script to check Qdrant collection directly via REST API
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def check_qdrant():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

    headers = {}
    if api_key:
        headers["api-key"] = api_key

    async with httpx.AsyncClient() as client:
        # Get collection info via REST API
        response = await client.get(
            f"{url}/collections/{collection_name}",
            headers=headers,
            timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            points_count = data["result"]["points_count"]
            print(f"Collection: {collection_name}")
            print(f"Points count: {points_count}")
            print(f"Vector size: {data['result']['config']['params']['vectors']['size']}")
            print(f"Distance: {data['result']['config']['params']['vectors']['distance']}")
            return points_count
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

if __name__ == "__main__":
    asyncio.run(check_qdrant())
