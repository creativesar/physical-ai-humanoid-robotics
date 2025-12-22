import os
import glob
import asyncio
import re
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentIngestor:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

        # MAXIMUM GRANULARITY CHUNKING PARAMETERS
        # Target: 30,000+ chunks from 37 files = ~811 chunks/file
        # Strategy: Aggressive small chunks with minimal filtering
        self.chunk_size = 100         # Small but not tiny (avoid too much noise)
        self.chunk_overlap = 30       # Good overlap for context
        self.min_chunk_size = 30      # Lower threshold to capture more content

        logger.info(f"ContentIngestor initialized with MAXIMUM GRANULARITY chunking")
        logger.info(f"  chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, min={self.min_chunk_size}")
        logger.info(f"  Target: 30,000+ chunks from 37 files (~811 chunks/file)")

    async def ingest_from_directory(self, docs_path: str, base_url: str = "/") -> Dict[str, Any]:
        """
        Ingest content from a directory of markdown files with MAXIMUM granularity chunking
        """
        if not os.path.exists(docs_path):
            raise ValueError(f"Docs path does not exist: {docs_path}")

        # Find all markdown files recursively
        markdown_files = []
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.md", recursive=True))
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.mdx", recursive=True))

        total_files = len(markdown_files)
        logger.info(f"Found {total_files} markdown files to process")

        print(f"\n{'='*80}")
        print(f"MAXIMUM GRANULARITY CONTENT INGESTION - TARGET: 30,000+ CHUNKS")
        print(f"{'='*80}")
        print(f"Files to process: {total_files}")
        print(f"Target chunks: 30,000+ (~{30000//total_files} per file)")
        print(f"Chunk size: {self.chunk_size} chars | Overlap: {self.chunk_overlap} chars | Min: {self.min_chunk_size} chars")
        print(f"{'='*80}\n")

        # Process each file
        total_chunks = 0
        total_files_processed = 0
        failed_files = []

        for idx, file_path in enumerate(markdown_files, 1):
            try:
                chunks_created = await self._process_file(file_path, base_url)
                total_chunks += chunks_created
                total_files_processed += 1

                progress_pct = (idx / total_files) * 100
                avg_per_file = total_chunks / total_files_processed if total_files_processed > 0 else 0

                logger.info(f"[{idx}/{total_files}] {file_path} -> {chunks_created} chunks (Total: {total_chunks})")
                print(f"[{idx}/{total_files}] ({progress_pct:>5.1f}%) {os.path.basename(file_path):<45} {chunks_created:>6} chunks | Total: {total_chunks:>7} | Avg: {avg_per_file:>6.1f}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")
                print(f"[ERROR] {os.path.basename(file_path)}: {str(e)}")
                failed_files.append({"file": file_path, "error": str(e)})

        print(f"\n{'='*80}")
        print(f"INGESTION COMPLETE")
        print(f"{'='*80}")

        return {
            "total_files_processed": total_files_processed,
            "total_chunks_created": total_chunks,
            "total_failed": len(failed_files),
            "failed": failed_files
        }

    async def _process_file(self, file_path: str, base_url: str) -> int:
        """
        Process a single markdown file with maximum granularity chunking
        Returns the number of chunks created
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata
        relative_path = self._get_relative_path(file_path)
        module_name = self._extract_module_name(relative_path)
        source_url = base_url + relative_path.replace('.md', '').replace('.mdx', '').replace('\\', '/')

        chunks_created = 0

        # Step 1: Clean the content but preserve structure
        cleaned_content = self._clean_markdown_preserve_structure(content)

        # Step 2: Split into semantic sections first (by headings, paragraphs, lists, etc.)
        sections = self._split_into_sections(cleaned_content)

        # Step 3: Apply recursive text splitter to each section
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple blank lines
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons
                ": ",      # Colons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Characters
            ]
        )

        for section_idx, section_text in enumerate(sections):
            # Skip empty sections
            if len(section_text.strip()) < self.min_chunk_size:
                continue

            # Extract heading context from section
            heading_context = self._extract_heading_from_section(section_text)

            # Split section into chunks
            section_chunks = text_splitter.split_text(section_text)

            # Index each chunk
            for chunk_text in section_chunks:
                # Apply minimum size filter
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                # Build metadata
                chunk_metadata = {
                    "chapter_id": f"{relative_path.replace('/', '_').replace('.md', '').replace('.mdx', '')}_c{chunks_created:05d}",
                    "section_title": heading_context or self._get_title_from_filename(file_path),
                    "source_file": relative_path,
                    "module": module_name,
                    "chunk_index": chunks_created,
                    "section_index": section_idx,
                    "chunk_size": len(chunk_text),
                }

                # Index the chunk
                try:
                    await self.rag_service.index_content(
                        content=chunk_text.strip(),
                        chapter_id=chunk_metadata["chapter_id"],
                        section_title=chunk_metadata["section_title"],
                        source_url=source_url,
                        metadata=chunk_metadata
                    )
                    chunks_created += 1
                except Exception as e:
                    logger.error(f"Error indexing chunk from {file_path}: {str(e)}")

        return chunks_created

    def _clean_markdown_preserve_structure(self, content: str) -> str:
        """
        Clean markdown but preserve structural elements that aid chunking
        """
        # Remove frontmatter (YAML between ---)
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Normalize whitespace but preserve paragraph breaks
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        return content

    def _split_into_sections(self, content: str) -> List[str]:
        """
        Split content into semantic sections for better chunking
        This creates more initial splits before fine-grained chunking
        """
        sections = []

        # Split by markdown headings first (# ## ### ####)
        heading_pattern = r'(^#{1,4}\s+.+$)'
        parts = re.split(heading_pattern, content, flags=re.MULTILINE)

        current_section = ""
        for part in parts:
            if not part.strip():
                continue

            # If it's a heading, start a new section
            if re.match(r'^#{1,4}\s+', part):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part + "\n"
            else:
                current_section += part

        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())

        # Further split large sections by blank lines
        refined_sections = []
        for section in sections:
            if len(section) > 1000:  # If section is large, split it further
                subsections = re.split(r'\n\n+', section)
                refined_sections.extend([s for s in subsections if s.strip()])
            else:
                refined_sections.append(section)

        # Split list items into individual sections
        final_sections = []
        for section in refined_sections:
            # Check if section contains bullet lists
            if re.search(r'^\s*[-*+]\s', section, re.MULTILINE):
                # Split by list items
                list_items = re.split(r'\n\s*[-*+]\s', section)
                final_sections.extend([item.strip() for item in list_items if item.strip()])
            # Check if section contains numbered lists
            elif re.search(r'^\s*\d+\.\s', section, re.MULTILINE):
                # Split by numbered items
                numbered_items = re.split(r'\n\s*\d+\.\s', section)
                final_sections.extend([item.strip() for item in numbered_items if item.strip()])
            else:
                final_sections.append(section)

        return final_sections if final_sections else [content]

    def _extract_heading_from_section(self, section_text: str) -> str:
        """
        Extract heading from section text if present
        """
        match = re.match(r'^(#{1,4})\s+(.+)$', section_text, re.MULTILINE)
        if match:
            return match.group(2).strip()
        return ""

    def _get_relative_path(self, file_path: str) -> str:
        """Extract relative path from docs directory"""
        if "docs" in file_path:
            docs_index = file_path.rfind("docs") + 5
            return file_path[docs_index:].replace('\\', '/')
        return os.path.basename(file_path)

    def _extract_module_name(self, relative_path: str) -> str:
        """Extract module name from path"""
        parts = relative_path.split('/')
        for part in parts:
            if part.startswith('module-') or part in ['assessments', 'getting-started', 'advanced']:
                return part
        return "general"

    def _get_title_from_filename(self, file_path: str) -> str:
        """Get title from filename"""
        filename = os.path.basename(file_path).replace('.md', '').replace('.mdx', '')
        return filename.replace('-', ' ').replace('_', ' ').title()

async def main():
    """Main function to run the content ingestor"""
    mistral_service = MistralService()
    qdrant_service = QdrantService()
    rag_service = RAGService(mistral_service, qdrant_service)

    ingestor = ContentIngestor(rag_service)
    await rag_service.initialize_collection()

    docs_path = os.getenv("DOCS_PATH", "../frontend/docs")
    results = await ingestor.ingest_from_directory(docs_path)

    logger.info(f"Ingestion completed. Files: {results['total_files_processed']}, Chunks: {results['total_chunks_created']}, Failed: {results['total_failed']}")

    if results['failed']:
        logger.error(f"Failed files: {results['failed']}")

if __name__ == "__main__":
    asyncio.run(main())
