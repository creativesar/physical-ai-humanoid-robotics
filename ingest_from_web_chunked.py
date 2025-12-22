#!/usr/bin/env python3
"""
Web scraper with automatic chunking for large documents
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlunparse
import logging
import tiktoken

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkedWebContentIngestor:
    def __init__(self, rag_service: RAGService, max_tokens: int = 6000):
        """
        Initialize with chunking support

        Args:
            rag_service: The RAG service for indexing
            max_tokens: Maximum tokens per chunk (default 6000 to stay well under 8192 limit)
        """
        self.rag_service = rag_service
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.max_tokens = max_tokens

        # Initialize tokenizer for counting tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("Could not load tiktoken encoder, using char count estimate")
            self.tokenizer = None

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def chunk_text(self, text: str, title: str) -> List[Dict[str, str]]:
        """
        Split text into chunks that fit within token limit

        Returns list of dicts with 'content' and 'chunk_index'
        """
        token_count = self.count_tokens(text)

        if token_count <= self.max_tokens:
            return [{'content': text, 'chunk_index': 0, 'total_chunks': 1}]

        # Need to chunk
        logger.info(f"Chunking document '{title}' ({token_count} tokens) into smaller pieces")

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds limit, split it further
            if para_tokens > self.max_tokens:
                # If we have accumulated content, save it first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                sentences = para.replace('. ', '.\n').split('\n')
                sentence_chunk = []
                sentence_tokens = 0

                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    if sentence_tokens + sent_tokens > self.max_tokens and sentence_chunk:
                        chunks.append(' '.join(sentence_chunk))
                        sentence_chunk = [sent]
                        sentence_tokens = sent_tokens
                    else:
                        sentence_chunk.append(sent)
                        sentence_tokens += sent_tokens

                if sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))

            # Normal case: paragraph fits
            elif current_tokens + para_tokens > self.max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f"Created {len(chunks)} chunks for '{title}'")

        # Return as list of dicts with metadata
        return [
            {
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]

    async def fetch_sitemap(self, sitemap_url: str, replace_domain: str = None) -> List[str]:
        """Fetch and parse sitemap XML to extract all URLs"""
        try:
            response = await self.client.get(sitemap_url)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

            if not urls:
                urls = [loc.text for loc in root.findall('.//loc')]

            if replace_domain:
                corrected_urls = []
                for url in urls:
                    parsed = urlparse(url)
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
        """Fetch and extract content from a single page"""
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find(class_='content') or soup.body

            if not main_content:
                logger.warning(f"No main content found for {url}")
                return None

            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else urlparse(url).path.strip('/')

            text_content = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)

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

    async def ingest_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Fetch and index content from a single URL with automatic chunking

        Returns list of indexed chunks
        """
        page_data = await self.fetch_page_content(url)

        if not page_data or not page_data['content']:
            logger.warning(f"Skipping {url} - no content extracted")
            return []

        # Chunk the content
        chunks = self.chunk_text(page_data['content'], page_data['title'])

        # Index each chunk
        results = []
        for chunk_data in chunks:
            try:
                # Create unique chapter_id for each chunk
                if chunk_data['total_chunks'] > 1:
                    chunk_chapter_id = f"{page_data['chapter_id']}_chunk_{chunk_data['chunk_index']}"
                    chunk_title = f"{page_data['title']} (Part {chunk_data['chunk_index'] + 1}/{chunk_data['total_chunks']})"
                else:
                    chunk_chapter_id = page_data['chapter_id']
                    chunk_title = page_data['title']

                result = await self.rag_service.index_content(
                    content=chunk_data['content'],
                    chapter_id=chunk_chapter_id,
                    section_title=chunk_title,
                    source_url=url
                )

                logger.info(f"Indexed: {chunk_title} ({len(chunk_data['content'])} chars)")

                results.append({
                    'url': url,
                    'title': chunk_title,
                    'chapter_id': chunk_chapter_id,
                    'content_length': len(chunk_data['content']),
                    'document_id': result['document_id'],
                    'chunk_index': chunk_data['chunk_index']
                })

            except Exception as e:
                logger.error(f"Error indexing chunk {chunk_data['chunk_index']} of {url}: {str(e)}")
                continue

        return results

    async def ingest_from_sitemap(self, sitemap_url: str, skip_urls: List[str] = None, replace_domain: str = None) -> Dict[str, Any]:
        """Ingest all content from a sitemap with chunking support"""
        skip_urls = skip_urls or []

        urls = await self.fetch_sitemap(sitemap_url, replace_domain=replace_domain)

        urls_to_process = []
        for url in urls:
            parsed_url = urlparse(url)
            skip = False
            for pattern in skip_urls:
                if parsed_url.path.rstrip('/') == pattern or parsed_url.path.rstrip('/') == pattern.rstrip('/'):
                    skip = True
                    break
            if not skip:
                urls_to_process.append(url)

        logger.info(f"Processing {len(urls_to_process)} URLs (skipping {len(urls) - len(urls_to_process)})")

        all_results = []
        failed_urls = []
        total_chunks = 0

        for i, url in enumerate(urls_to_process, 1):
            logger.info(f"Processing {i}/{len(urls_to_process)}: {url}")

            results = await self.ingest_url(url)

            if results:
                all_results.extend(results)
                total_chunks += len(results)
            else:
                failed_urls.append(url)

            await asyncio.sleep(0.5)

        return {
            'total_urls': len(urls),
            'urls_processed': len(urls_to_process) - len(failed_urls),
            'total_chunks_created': total_chunks,
            'failed': len(failed_urls),
            'skipped': len(urls) - len(urls_to_process),
            'successful': all_results,
            'failed_urls': failed_urls
        }


async def main():
    """Main function"""
    logger.info("Starting chunked web content ingestion...")

    mistral_service = MistralService()
    qdrant_service = QdrantService()
    rag_service = RAGService(mistral_service, qdrant_service)

    await rag_service.initialize_collection()

    ingestor = ChunkedWebContentIngestor(rag_service, max_tokens=6000)

    try:
        sitemap_url = "https://physical-ai-humanoid-robotics-seven-red.vercel.app/sitemap.xml"
        actual_domain = "physical-ai-humanoid-robotics-seven-red.vercel.app"
        skip_patterns = ['/about', '/contact', '/LayoutWrapper', '/markdown-page',
                        '/our-vision', '/privacy', '/signin', '/signup', '/terms']

        results = await ingestor.ingest_from_sitemap(
            sitemap_url,
            skip_urls=skip_patterns,
            replace_domain=actual_domain
        )

        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total URLs in sitemap: {results['total_urls']}")
        logger.info(f"URLs successfully processed: {results['urls_processed']}")
        logger.info(f"Total chunks created: {results['total_chunks_created']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Skipped: {results['skipped']}")

        if results['failed_urls']:
            logger.warning(f"Failed URLs: {results['failed_urls']}")

        # Check final point count
        try:
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
