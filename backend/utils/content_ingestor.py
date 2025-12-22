import os
import glob
import asyncio
import re
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from services.mistral_service import MistralService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentIngestor:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

        # Configure markdown header splitter to split by all heading levels
        self.headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]

        # ULTRA FINE-GRAINED CHUNKING PARAMETERS (target: 25,000 chunks from 37 files)
        # Average file size: ~35,000 chars -> need ~676 chunks/file
        # With 50-char chunks: 35,000/50 = 700 chunks/file ✓
        self.chunk_size = 50  # Very small chunks for maximum granularity
        self.chunk_overlap = 15  # Overlap to maintain context
        self.min_chunk_size = 20  # Skip only very tiny chunks

        logger.info(f"ContentIngestor initialized with ULTRA FINE-GRAINED chunking")
        logger.info(f"  chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, min={self.min_chunk_size}")
        logger.info(f"  Target: ~25,000 chunks from 37 files (~676 chunks/file)")

    async def ingest_from_directory(self, docs_path: str, base_url: str = "/") -> Dict[str, Any]:
        """
        Ingest content from a directory of markdown files with ULTRA fine-grained chunking
        """
        if not os.path.exists(docs_path):
            raise ValueError(f"Docs path does not exist: {docs_path}")

        # Find all markdown files recursively
        markdown_files = []
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.md", recursive=True))
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.mdx", recursive=True))

        logger.info(f"Found {len(markdown_files)} markdown files to process")
        print(f"\n{'='*70}")
        print(f"ULTRA FINE-GRAINED CONTENT INGESTION")
        print(f"{'='*70}")
        print(f"[INFO] Found {len(markdown_files)} markdown files to process")
        print(f"[INFO] Target: ~25,000 total chunks (~676 per file)")
        print(f"[INFO] Chunk size: {self.chunk_size} chars, Overlap: {self.chunk_overlap} chars")
        print(f"{'='*70}\n")

        # Process each file
        total_chunks = 0
        total_files_processed = 0
        failed_files = []

        for idx, file_path in enumerate(markdown_files, 1):
            try:
                chunks_created = await self._process_file(file_path, base_url)
                total_chunks += chunks_created
                total_files_processed += 1

                progress_pct = (idx / len(markdown_files)) * 100
                logger.info(f"[{idx}/{len(markdown_files)}] ({progress_pct:.1f}%) {file_path} -> {chunks_created} chunks (Total: {total_chunks})")
                print(f"[{idx}/{len(markdown_files)}] ({progress_pct:.1f}%) {os.path.basename(file_path):<40} -> {chunks_created:>5} chunks (Running total: {total_chunks:>6})")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")
                print(f"[ERROR] Failed: {os.path.basename(file_path)} - {str(e)}")
                failed_files.append({"file": file_path, "error": str(e)})

        print(f"\n{'='*70}")
        print(f"INGESTION COMPLETE")
        print(f"{'='*70}")

        return {
            "total_files_processed": total_files_processed,
            "total_chunks_created": total_chunks,
            "total_failed": len(failed_files),
            "failed": failed_files
        }

    async def _process_file(self, file_path: str, base_url: str) -> int:
        """
        Process a single markdown file with ULTRA fine-grained chunking
        Returns the number of chunks created
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata from file path
        relative_path = self._get_relative_path(file_path)
        module_name = self._extract_module_name(relative_path)

        # Create source URL
        source_url = base_url + relative_path.replace('.md', '').replace('.mdx', '').replace('\\', '/')

        chunks_created = 0

        # Strategy 1: Split by markdown headers first (preserve hierarchy)
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep headers for context
        )

        try:
            header_splits = header_splitter.split_text(content)
        except Exception as e:
            logger.warning(f"Header splitting failed for {file_path}, using full content: {str(e)}")
            # Create a fallback document-like object
            class Document:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata
            header_splits = [Document(content, {})]

        # Strategy 2: Split each header section into ULTRA fine-grained chunks
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""],
            length_function=len,
        )

        for header_idx, header_doc in enumerate(header_splits):
            # Get the text content
            text = header_doc.page_content if hasattr(header_doc, 'page_content') else str(header_doc)
            header_metadata = header_doc.metadata if hasattr(header_doc, 'metadata') else {}

            # Strategy 3: Additional preprocessing to maximize chunks
            # Split by special patterns (lists, code blocks, tables)
            preprocessed_segments = self._preprocess_for_maximum_chunks(text)

            for segment_idx, segment in enumerate(preprocessed_segments):
                # Skip very small segments
                if len(segment.strip()) < self.min_chunk_size:
                    continue

                # Split each segment into ultra-fine chunks
                fine_chunks = recursive_splitter.split_text(segment)

                # Index each fine-grained chunk
                for chunk_idx, chunk_text in enumerate(fine_chunks):
                    # Skip very small chunks
                    if len(chunk_text.strip()) < self.min_chunk_size:
                        continue

                    # Build rich metadata
                    chunk_metadata = self._build_chunk_metadata(
                        file_path=file_path,
                        relative_path=relative_path,
                        module_name=module_name,
                        source_url=source_url,
                        header_metadata=header_metadata,
                        chunk_index=chunks_created,  # Global chunk index for this file
                        chunk_text=chunk_text,
                        header_idx=header_idx,
                        segment_idx=segment_idx
                    )

                    # Index the chunk
                    try:
                        await self.rag_service.index_content(
                            content=chunk_text,
                            chapter_id=chunk_metadata["chapter_id"],
                            section_title=chunk_metadata["section_title"],
                            source_url=source_url,
                            metadata=chunk_metadata
                        )
                        chunks_created += 1
                    except Exception as e:
                        logger.error(f"Error indexing chunk from {file_path}: {str(e)}")
                        # Continue processing other chunks

        return chunks_created

    def _preprocess_for_maximum_chunks(self, text: str) -> List[str]:
        """
        Preprocess text to create maximum possible chunks by splitting on special patterns
        """
        segments = []

        # Split by multiple newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)

        for para in paragraphs:
            if not para.strip():
                continue

            # Check for list items (bullets, numbers)
            if re.match(r'^[\s]*[-*+•]\s', para) or re.match(r'^[\s]*\d+\.\s', para):
                # Split list into individual items
                list_items = re.split(r'\n[\s]*(?:[-*+•]|\d+\.)\s', para)
                segments.extend([item.strip() for item in list_items if item.strip()])

            # Check for code blocks
            elif '```' in para:
                # Split code blocks from text
                code_parts = re.split(r'```[\w]*\n?', para)
                segments.extend([part.strip() for part in code_parts if part.strip()])

            # Check for tables (markdown tables with |)
            elif '|' in para and para.count('|') >= 3:
                # Split table rows
                table_rows = para.split('\n')
                segments.extend([row.strip() for row in table_rows if row.strip() and '|' in row])

            # Regular paragraph - split by sentences
            else:
                # Split by sentence-ending punctuation
                sentences = re.split(r'([.!?]+[\s\n]+)', para)
                current_sentence = ""
                for i, part in enumerate(sentences):
                    current_sentence += part
                    # If this is punctuation followed by space, it's end of sentence
                    if re.match(r'[.!?]+[\s\n]+', part):
                        if current_sentence.strip():
                            segments.append(current_sentence.strip())
                        current_sentence = ""
                # Add remaining text
                if current_sentence.strip():
                    segments.append(current_sentence.strip())

        # If preprocessing didn't split much, return original as single segment
        if not segments:
            segments = [text]

        return segments

    def _get_relative_path(self, file_path: str) -> str:
        """
        Extract relative path from docs directory
        """
        if "docs" in file_path:
            docs_index = file_path.rfind("docs") + 5  # +5 to skip "docs/"
            relative_path = file_path[docs_index:].replace('\\', '/')
        else:
            relative_path = os.path.basename(file_path)
        return relative_path

    def _extract_module_name(self, relative_path: str) -> str:
        """
        Extract module name from path (e.g., "module-1", "module-2", "assessments")
        """
        parts = relative_path.split('/')
        for part in parts:
            if part.startswith('module-') or part in ['assessments', 'getting-started', 'advanced']:
                return part
        return "general"

    def _build_chunk_metadata(
        self,
        file_path: str,
        relative_path: str,
        module_name: str,
        source_url: str,
        header_metadata: Dict[str, Any],
        chunk_index: int,
        chunk_text: str,
        header_idx: int = 0,
        segment_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Build rich metadata for each chunk
        """
        # Create a descriptive chapter_id
        chapter_id = relative_path.replace('/', '_').replace('\\', '_').replace('.md', '').replace('.mdx', '')

        # Create section title from headers or filename
        section_title = self._build_section_title(header_metadata, file_path)

        # Build complete metadata
        metadata = {
            "chapter_id": f"{chapter_id}_c{chunk_index:04d}",  # c0001, c0002, etc.
            "section_title": section_title,
            "source_file": relative_path,
            "module": module_name,
            "chunk_index": chunk_index,
            "chunk_size": len(chunk_text),
            "header_section": header_idx,
            "segment": segment_idx,
        }

        # Add heading hierarchy from markdown headers
        if header_metadata:
            for key, value in header_metadata.items():
                if key.startswith('H'):
                    metadata[key.lower()] = value

        return metadata

    def _build_section_title(self, header_metadata: Dict[str, Any], file_path: str) -> str:
        """
        Build a descriptive section title from header metadata or filename
        """
        # Try to build from headers (most specific to least specific)
        for header_level in ['H4', 'H3', 'H2', 'H1']:
            if header_level in header_metadata:
                return header_metadata[header_level]

        # Fall back to filename
        filename = os.path.basename(file_path).replace('.md', '').replace('.mdx', '')
        # Clean up filename (replace dashes/underscores with spaces, title case)
        return filename.replace('-', ' ').replace('_', ' ').title()

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

    logger.info(f"Ingestion completed. Files: {results['total_files_processed']}, Chunks: {results['total_chunks_created']}, Failed: {results['total_failed']}")

    if results['failed']:
        logger.error(f"Failed files: {results['failed']}")

if __name__ == "__main__":
    asyncio.run(main())
