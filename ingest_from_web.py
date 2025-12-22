#!/usr/bin/env python3
"""
Web scraper to ingest content from the deployed site using sitemap
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebContentIngestor:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def fetch_sitemap(self, sitemap_url: str, replace_domain: str = None) -> List[str]:
        """
        Fetch and parse sitemap XML to extract all URLs

        Args:
            sitemap_url: URL of the sitemap
            replace_domain: If provided, replace the domain in sitemap URLs with this domain
        """
        try:
            response = await self.client.get(sitemap_url)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            # Extract URLs from sitemap (handle namespace)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

            # If no namespace found, try without it
            if not urls:
                urls = [loc.text for loc in root.findall('.//loc')]

            # Replace domain if requested
            if replace_domain:
                from urllib.parse import urlparse, urlunparse
                corrected_urls = []
                for url in urls:
                    parsed = urlparse(url)
                    # Replace the netloc (domain) but keep the path
                    new_parsed = parsed._replace(netloc=replace_domain)
                    corrected_urls.append(urlunparse(new_parsed))
                urls = corrected_urls
                logger.info(f"Replaced domain with {replace_domain}")

            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls

        except Exception as e:
            logger.error(f"Error fetching sitemap: {str(e)}")
            raise

    async def fetch_page_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract content from a single page
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script, style, nav, footer, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()

            # Try to find main content area (adjust selectors based on your site structure)
            main_content = soup.find('main') or soup.find('article') or soup.find(class_='content') or soup.body

            if not main_content:
                logger.warning(f"No main content found for {url}")
                return None

            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else urlparse(url).path.strip('/')

            # Extract all text content
            text_content = main_content.get_text(separator='\n', strip=True)

            # Clean up: remove multiple newlines and extra spaces
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)

            # Extract metadata
            path = urlparse(url).path
            chapter_id = path.replace('/', '_').strip('_') or 'home'

            return {
                'url': url,
                'title': title_text,
                'content': clean_content,
                'chapter_id': chapter_id,
                'path': path
            }

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch and index content from a single URL
        """
        page_data = await self.fetch_page_content(url)

        if not page_data or not page_data['content']:
            logger.warning(f"Skipping {url} - no content extracted")
            return None

        # Index the content
        try:
            result = await self.rag_service.index_content(
                content=page_data['content'],
                chapter_id=page_data['chapter_id'],
                section_title=page_data['title'],
                source_url=url
            )

            logger.info(f"Indexed: {page_data['title']} ({len(page_data['content'])} chars)")

            return {
                'url': url,
                'title': page_data['title'],
                'chapter_id': page_data['chapter_id'],
                'content_length': len(page_data['content']),
                'document_id': result['document_id']
            }

        except Exception as e:
            logger.error(f"Error indexing {url}: {str(e)}")
            return None

    async def ingest_from_sitemap(self, sitemap_url: str, skip_urls: List[str] = None, replace_domain: str = None) -> Dict[str, Any]:
        """
        Ingest all content from a sitemap

        Args:
            sitemap_url: URL of the sitemap
            skip_urls: List of URL patterns to skip
            replace_domain: If provided, replace the domain in sitemap URLs with this domain
        """
        skip_urls = skip_urls or []

        # Fetch all URLs from sitemap
        urls = await self.fetch_sitemap(sitemap_url, replace_domain=replace_domain)

        # Filter URLs if needed (check if URL path contains any skip pattern)
        urls_to_process = []
        for url in urls:
            parsed_url = urlparse(url)
            # Check if the path matches any skip pattern
            skip = False
            for pattern in skip_urls:
                if parsed_url.path.rstrip('/') == pattern or parsed_url.path.rstrip('/') == pattern.rstrip('/'):
                    skip = True
                    break
            if not skip:
                urls_to_process.append(url)

        logger.info(f"Processing {len(urls_to_process)} URLs (skipping {len(skip_urls)})")

        # Process each URL
        results = []
        failed = []

        for i, url in enumerate(urls_to_process, 1):
            logger.info(f"Processing {i}/{len(urls_to_process)}: {url}")

            result = await self.ingest_url(url)

            if result:
                results.append(result)
            else:
                failed.append(url)

            # Small delay to be polite to the server
            await asyncio.sleep(0.5)

        return {
            'total_urls': len(urls),
            'processed': len(results),
            'failed': len(failed),
            'skipped': len(skip_urls),
            'successful': results,
            'failed_urls': failed
        }


async def main():
    """
    Main function to run the web content ingestor
    """
    logger.info("Starting web content ingestion...")

    # Initialize services
    mistral_service = MistralService()
    qdrant_service = QdrantService()
    rag_service = RAGService(mistral_service, qdrant_service)

    # Initialize the Qdrant collection
    await rag_service.initialize_collection()

    # Initialize web ingestor
    ingestor = WebContentIngestor(rag_service)

    try:
        # Your sitemap URL
        sitemap_url = "https://physical-ai-humanoid-robotics-seven-red.vercel.app/sitemap.xml"

        # The actual domain where content is hosted (replace sitemap URLs with this)
        actual_domain = "physical-ai-humanoid-robotics-seven-red.vercel.app"

        # Optional: Skip non-content pages (navigation, auth pages, etc.) - using patterns
        skip_patterns = ['/about', '/contact', '/LayoutWrapper', '/markdown-page',
                        '/our-vision', '/privacy', '/signin', '/signup', '/terms']

        # Ingest content with domain replacement
        results = await ingestor.ingest_from_sitemap(
            sitemap_url,
            skip_urls=skip_patterns,
            replace_domain=actual_domain
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total URLs in sitemap: {results['total_urls']}")
        logger.info(f"Successfully processed: {results['processed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Skipped: {results['skipped']}")

        if results['failed_urls']:
            logger.warning(f"Failed URLs: {results['failed_urls']}")

        # Check final point count using REST API to avoid Pydantic issues
        try:
            import httpx
            import os
            async with httpx.AsyncClient() as client:
                url = os.getenv("QDRANT_URL", "http://localhost:6333")
                api_key = os.getenv("QDRANT_API_KEY")
                collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

                headers = {}
                if api_key:
                    headers["api-key"] = api_key

                response = await client.get(
                    f"{url}/collections/{collection_name}",
                    headers=headers,
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    point_count = data["result"]["points_count"]
                    logger.info(f"Total points in Qdrant: {point_count}")
                else:
                    logger.warning(f"Could not fetch point count: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not fetch final point count: {e}")

    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
