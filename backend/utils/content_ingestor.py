import os
import glob
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
import logging

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentIngestor:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    async def ingest_from_directory(self, docs_path: str, base_url: str = "/docs/") -> Dict[str, Any]:
        """
        Ingest content from a directory of markdown files
        """
        if not os.path.exists(docs_path):
            raise ValueError(f"Docs path does not exist: {docs_path}")

        # Find all markdown files
        markdown_files = []
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.md", recursive=True))
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.mdx", recursive=True))

        logger.info(f"Found {len(markdown_files)} markdown files to process")

        # Process each file
        results = []
        failed_files = []

        for file_path in markdown_files:
            try:
                result = await self._process_file(file_path, base_url)
                if result:
                    results.append(result)
                    logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")
                failed_files.append({"file": file_path, "error": str(e)})

        return {
            "total_processed": len(results),
            "total_failed": len(failed_files),
            "successful": results,
            "failed": failed_files
        }

    async def _process_file(self, file_path: str, base_url: str) -> Dict[str, Any]:
        """
        Process a single markdown file and extract content for indexing
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract chapter/section info from file path
        relative_path = os.path.relpath(file_path, os.path.dirname(file_path.split("docs")[0] + "docs"))

        # Clean up the path to create a proper chapter ID
        chapter_id = relative_path.replace('/', '_').replace('\\', '_').replace('.md', '').replace('.mdx', '')
        section_title = os.path.basename(file_path).replace('.md', '').replace('.mdx', '')

        # Convert markdown to plain text for better indexing
        plain_text = self._markdown_to_text(content)

        # Create source URL
        source_url = base_url + relative_path.replace('.md', '').replace('.mdx', '').replace('\\', '/')

        # Index the content
        index_result = await self.rag_service.index_content(
            content=plain_text,
            chapter_id=chapter_id,
            section_title=section_title,
            source_url=source_url
        )

        return {
            "file_path": file_path,
            "chapter_id": chapter_id,
            "section_title": section_title,
            "source_url": source_url,
            "indexed_document_id": index_result["document_id"]
        }

    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert markdown content to plain text, preserving important content
        """
        try:
            # Convert markdown to HTML first
            html = markdown.markdown(markdown_content)

            # Parse HTML and extract text
            soup = BeautifulSoup(html, 'html.parser')

            # Remove code blocks and other elements that might not be relevant for RAG
            for code_block in soup.find_all(['code', 'pre']):
                code_block.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up extra whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)

            return clean_text
        except Exception as e:
            logger.warning(f"Error converting markdown to text: {str(e)}, returning original content")
            return markdown_content

    async def chunk_and_ingest_content(self, content: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Chunk large content into smaller pieces for better indexing
        """
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)

        results = []
        for i, chunk in enumerate(chunks):
            # Index each chunk separately
            result = await self.rag_service.index_content(
                content=chunk,
                chapter_id=f"chunk_{i}",
                section_title=f"Chunk {i}",
                source_url=f"#chunk-{i}"
            )
            results.append(result)

        return results

async def main():
    """
    Main function to run the content ingestor
    """
    # Initialize services
    mistral_service = MistralService()
    qdrant_service = QdrantService()
    rag_service = RAGService(mistral_service, qdrant_service)

    # Initialize content ingestor
    ingestor = ContentIngestor(rag_service)

    # Initialize the Qdrant collection
    await rag_service.initialize_collection()

    # Get docs path from environment or use default
    docs_path = os.getenv("DOCS_PATH", "../frontend/docs")

    # Ingest content
    results = await ingestor.ingest_from_directory(docs_path)

    logger.info(f"Ingestion completed. Processed: {results['total_processed']}, Failed: {results['total_failed']}")

    if results['failed']:
        logger.error(f"Failed files: {results['failed']}")

if __name__ == "__main__":
    asyncio.run(main())