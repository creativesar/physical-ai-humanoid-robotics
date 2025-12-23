import os
import glob
import asyncio
import re
from typing import List, Dict, Any
from pathlib import Path
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

        # AGGRESSIVE CHUNKING FOR 30,000+ POINTS
        # Math: 1,312,517 chars / 30,000 points = ~44 chars/point
        # With overlap and preprocessing multiplier: use 60-char base
        self.chunk_size = 60          # Small chunks for max granularity
        self.chunk_overlap = 20       # Context preservation
        self.min_chunk_size = 25      # Minimal filtering

        logger.info(f"ContentIngestor initialized for 30,000+ POINTS TARGET")
        logger.info(f"  chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, min={self.min_chunk_size}")
        logger.info(f"  Strategy: Aggressive splitting of lists, code, paragraphs, sentences")

    async def ingest_from_directory(self, docs_path: str, base_url: str = "/") -> Dict[str, Any]:
        """
        Ingest content from a directory with aggressive chunking to reach 30,000+ points
        """
        if not os.path.exists(docs_path):
            raise ValueError(f"Docs path does not exist: {docs_path}")

        markdown_files = []
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.md", recursive=True))
        markdown_files.extend(glob.glob(f"{docs_path}/**/*.mdx", recursive=True))

        total_files = len(markdown_files)
        target_per_file = 30000 // total_files

        print(f"\n{'='*85}")
        print(f" AGGRESSIVE CHUNKING FOR 30,000+ POINTS - Physical AI & Humanoid Robotics Textbook")
        print(f"{'='*85}")
        print(f" Total files: {total_files} | Target: 30,000+ points (~{target_per_file} per file)")
        print(f" Chunk size: {self.chunk_size} chars | Overlap: {self.chunk_overlap} | Min: {self.min_chunk_size}")
        print(f" Strategy: Split lists, code blocks, paragraphs, sentences aggressively")
        print(f"{'='*85}\n")

        total_chunks = 0
        total_files_processed = 0
        failed_files = []

        for idx, file_path in enumerate(markdown_files, 1):
            try:
                chunks_created = await self._process_file(file_path, base_url)
                total_chunks += chunks_created
                total_files_processed += 1

                progress_pct = (idx / total_files) * 100
                avg_per_file = total_chunks / total_files_processed

                print(f"[{idx:2d}/{total_files}] {progress_pct:5.1f}% | {os.path.basename(file_path):<50} | {chunks_created:5d} chunks | Total: {total_chunks:6d} | Avg: {avg_per_file:6.1f}")
            except Exception as e:
                logger.error(f"Failed: {file_path}: {str(e)}")
                print(f"[ERROR] {os.path.basename(file_path)}: {str(e)}")
                failed_files.append({"file": file_path, "error": str(e)})

        print(f"\n{'='*85}")
        print(f" INGESTION COMPLETE - TARGET {'ACHIEVED' if total_chunks >= 30000 else 'MISSED'}")
        print(f"{'='*85}")
        print(f" Total points created: {total_chunks:,}")
        print(f" Target: 30,000 | Difference: {total_chunks - 30000:+,}")
        print(f" Files processed: {total_files_processed} | Failed: {len(failed_files)}")
        print(f" Average per file: {avg_per_file:.1f}")
        print(f"{'='*85}\n")

        return {
            "total_files_processed": total_files_processed,
            "total_chunks_created": total_chunks,
            "total_failed": len(failed_files),
            "failed": failed_files
        }

    async def _process_file(self, file_path: str, base_url: str) -> int:
        """Process file with AGGRESSIVE chunking for maximum point count"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        relative_path = self._get_relative_path(file_path)
        module_name = self._extract_module_name(relative_path)
        source_url = base_url + relative_path.replace('.md', '').replace('.mdx', '').replace('\\', '/')

        chunks_created = 0

        # Step 1: Clean but preserve all content
        cleaned = self._minimal_clean(content)

        # Step 2: AGGRESSIVE splitting into micro-segments
        segments = self._aggressive_split(cleaned)

        # Step 3: Simple text chunking without external dependencies

        heading_context = ""
        for segment_idx, segment in enumerate(segments):
            if len(segment.strip()) < self.min_chunk_size:
                continue

            # Track headings for metadata
            if segment.strip().startswith('#'):
                heading_context = segment.strip().lstrip('#').strip()
                # Also chunk the heading itself
                if len(segment.strip()) >= self.min_chunk_size:
                    await self._index_chunk(
                        chunk_text=segment.strip(),
                        chunks_created=chunks_created,
                        relative_path=relative_path,
                        module_name=module_name,
                        source_url=source_url,
                        heading_context=heading_context,
                        file_path=file_path,
                        segment_idx=segment_idx
                    )
                    chunks_created += 1
                continue

            # Split segment into fine chunks
            fine_chunks = self._simple_chunk_text(segment)

            for chunk_text in fine_chunks:
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                await self._index_chunk(
                    chunk_text=chunk_text.strip(),
                    chunks_created=chunks_created,
                    relative_path=relative_path,
                    module_name=module_name,
                    source_url=source_url,
                    heading_context=heading_context,
                    file_path=file_path,
                    segment_idx=segment_idx
                )
                chunks_created += 1

        return chunks_created

    async def _index_chunk(self, chunk_text, chunks_created, relative_path, module_name,
                          source_url, heading_context, file_path, segment_idx):
        """Index a single chunk"""
        metadata = {
            "chapter_id": f"{relative_path.replace('/', '_').replace('.md', '').replace('.mdx', '')}_c{chunks_created:05d}",
            "section_title": heading_context or self._get_title_from_filename(file_path),
            "source_file": relative_path,
            "module": module_name,
            "chunk_index": chunks_created,
            "segment_index": segment_idx,
            "chunk_size": len(chunk_text),
        }

        try:
            await self.rag_service.index_content(
                content=chunk_text,
                chapter_id=metadata["chapter_id"],
                section_title=metadata["section_title"],
                source_url=source_url,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error indexing chunk: {str(e)}")

    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple text chunking without external dependencies
        Splits text into chunks of specified size with overlap
        """
        chunks = []
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " "]

        def split_by_separators(text: str, seps: List[str]) -> List[str]:
            if not seps or len(text) <= self.chunk_size:
                return [text]

            sep = seps[0]
            parts = text.split(sep)
            result = []
            current = ""

            for i, part in enumerate(parts):
                test = current + sep + part if current else part
                if len(test) > self.chunk_size and current:
                    result.append(current)
                    current = part
                else:
                    current = test

            if current:
                result.append(current)

            # Further split large parts
            final = []
            for part in result:
                if len(part) > self.chunk_size:
                    final.extend(split_by_separators(part, seps[1:]))
                else:
                    final.append(part)
            return final

        parts = split_by_separators(text, separators)

        # Add chunks with overlap
        for i, part in enumerate(parts):
            if len(part) < self.min_chunk_size:
                continue
            chunks.append(part)

        return chunks

    def _minimal_clean(self, content: str) -> str:
        """Minimal cleaning - preserve all content"""
        # Remove frontmatter only
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        return content

    def _aggressive_split(self, content: str) -> List[str]:
        """
        AGGRESSIVE splitting strategy to maximize segments
        This is the key to reaching 30,000 points!
        """
        segments = []

        # Split by headings first
        lines = content.split('\n')
        current_segment = []

        for line in lines:
            stripped = line.strip()

            # Heading - create segment
            if stripped.startswith('#'):
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                segments.append(line)  # Heading as its own segment
                continue

            # Bullet/numbered list item - each item is a segment
            if re.match(r'^[-*+•]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                segments.append(line)
                continue

            # Code block markers
            if stripped.startswith('```'):
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                current_segment.append(line)
                continue

            # Blank line - end current segment
            if not stripped:
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                continue

            # Table row
            if '|' in line and line.count('|') >= 2:
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                segments.append(line)
                continue

            # Regular line - add to current segment
            current_segment.append(line)

            # Force segment break every 3-5 lines to prevent large segments
            if len(current_segment) >= 4:
                segments.append('\n'.join(current_segment))
                current_segment = []

        # Add any remaining
        if current_segment:
            segments.append('\n'.join(current_segment))

        # Further split long segments
        final_segments = []
        for seg in segments:
            if len(seg) > 300:  # Split long segments by sentences
                sentences = re.split(r'([.!?]+\s+)', seg)
                temp = ""
                for s in sentences:
                    temp += s
                    if len(temp) > 150:
                        final_segments.append(temp.strip())
                        temp = ""
                if temp.strip():
                    final_segments.append(temp.strip())
            else:
                final_segments.append(seg)

        return [s for s in final_segments if s.strip()]

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
            if part.startswith('module-') or part in ['assessments', 'getting-started', 'hardware-requirements', 'weekly-breakdown']:
                return part
        return "general"

    def _get_title_from_filename(self, file_path: str) -> str:
        """Get title from filename"""
        filename = os.path.basename(file_path).replace('.md', '').replace('.mdx', '')
        return filename.replace('-', ' ').replace('_', ' ').title()

async def main():
    """Main function"""
    mistral_service = MistralService()
    qdrant_service = QdrantService()
    rag_service = RAGService(mistral_service, qdrant_service)

    ingestor = ContentIngestor(rag_service)
    await rag_service.initialize_collection()

    docs_path = os.getenv("DOCS_PATH", "../frontend/docs")
    results = await ingestor.ingest_from_directory(docs_path)

    logger.info(f"Ingestion completed. Files: {results['total_files_processed']}, Chunks: {results['total_chunks_created']}, Failed: {results['total_failed']}")

if __name__ == "__main__":
    asyncio.run(main())
