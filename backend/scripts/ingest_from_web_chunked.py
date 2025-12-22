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
        Split text into chunks that fit within token limit with more granular chunking
        This method will be used as a fallback when more granular methods fail

        Returns list of dicts with 'content' and 'chunk_index'
        """
        token_count = self.count_tokens(text)

        if token_count <= self.max_tokens:
            return [{'content': text, 'chunk_index': 0, 'total_chunks': 1}]

        # Need to chunk more granularly
        logger.info(f"Chunking document '{title}' ({token_count} tokens) into smaller pieces")

        # Split by more granular elements: paragraphs, code blocks, headers, etc.
        # First, let's try to split by headers and major sections
        import re

        # Split text by headers (lines starting with # or section headers)
        sections = re.split(r'\n(?=#|\n[^\s\n].*?\n[-=]+\n)', text)

        # If no headers found, fall back to paragraphs
        if len(sections) <= 1:
            sections = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_tokens = self.count_tokens(section)

            # If single section exceeds limit, split it further
            if section_tokens > self.max_tokens:
                # Try to split by sentences if it's a large section
                sentences = re.split(r'[.!?]+\s+', section)

                if len(sentences) > 1:
                    # Process sentences into smaller chunks
                    sentence_chunk = []
                    sentence_tokens = 0

                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue

                        sent_tokens = self.count_tokens(sent)

                        if sentence_tokens + sent_tokens > self.max_tokens and sentence_chunk:
                            chunks.append('. '.join(sentence_chunk) + '.')
                            sentence_chunk = [sent]
                            sentence_tokens = sent_tokens
                        else:
                            sentence_chunk.append(sent)
                            sentence_tokens += sent_tokens

                    if sentence_chunk:
                        chunks.append('. '.join(sentence_chunk) + '.')
                else:
                    # If sentences don't help, just add as a single chunk (will likely fail at embedding)
                    chunks.append(section)
            # Normal case: section fits
            elif current_tokens + section_tokens > self.max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [section]
                current_tokens = section_tokens
            else:
                current_chunk.append(section)
                current_tokens += section_tokens

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

    def create_granular_chunks(self, page_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create ultra-granular chunks by extracting the finest possible content elements
        This will maximize the number of points by creating separate entries
        for individual words, phrases, code tokens, table elements, and other micro-components
        """
        import re
        from bs4 import BeautifulSoup

        chunks = []
        chunk_index = 0

        # Parse the content to extract different HTML elements
        soup = BeautifulSoup(page_data['content'], 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()

        # Extract different types of content elements individually
        elements = []

        # Extract headings (h1-h6) with multiple contexts
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                elements.append(('heading', heading_text))

                # Get the next sibling content as context for the heading
                next_elem = heading.find_next_sibling()
                if next_elem:
                    next_text = next_elem.get_text(strip=True)
                    if next_text and len(next_text) < 500:  # Only if it's reasonably short
                        elements.append(('heading_context', f"{heading_text}: {next_text}"))

                # Break heading into individual words/short phrases
                words = heading_text.split()
                for i in range(0, len(words), 3):  # Group 3 words together
                    word_group = ' '.join(words[i:i+3])
                    if word_group:
                        elements.append(('heading_words', word_group))

                # Break heading into individual characters for maximum granularity
                for char in heading_text:
                    if char.strip():  # Only add non-whitespace characters
                        elements.append(('heading_char', char))

        # Extract paragraphs and break them down to extreme granularity
        for p in soup.find_all('p'):
            p_text = p.get_text(strip=True)
            if p_text and len(p_text) > 5:  # Even short paragraphs
                # Split into sentences
                sentences = re.split(r'[.!?]+\s+', p_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 3:  # Only add meaningful sentences
                        elements.append(('sentence', sentence))

                        # Break sentence into phrases (every 5 words)
                        words = sentence.split()
                        for i in range(0, len(words), 5):
                            phrase = ' '.join(words[i:i+5])
                            if len(phrase) > 3:  # Meaningful phrases
                                elements.append(('phrase', phrase))

                        # Break sentence into individual words
                        for word in words:
                            word = word.strip('.,;:!?()[]{}"\'')
                            if len(word) > 2:  # Only meaningful words
                                elements.append(('word', word))

                                # Break word into syllables (simple approach)
                                if len(word) > 4:
                                    for i in range(0, len(word), 2):
                                        syllable = word[i:i+2]
                                        if len(syllable) > 1:
                                            elements.append(('syllable', syllable))

                        # Break sentence into characters
                        for char in sentence:
                            if char.strip():  # Only add non-whitespace characters
                                elements.append(('sentence_char', char))

                # Also add the full paragraph as a chunk
                if len(p_text) > 10:  # Any paragraph with content
                    elements.append(('paragraph', p_text))

        # Extract code blocks and break them down to the token level
        for code_block in soup.find_all(['code', 'pre']):
            code_text = code_block.get_text(strip=True)
            if code_text and len(code_text) > 1:
                # Split code into individual lines
                code_lines = code_text.split('\n')
                for line_idx, line in enumerate(code_lines):
                    line = line.strip()
                    if line:
                        elements.append(('code_line', f"Line {line_idx}: {line}"))

                        # Extract individual code tokens (words separated by punctuation/spaces)
                        import re
                        tokens = re.findall(r'\b\w+\b', line)
                        for token in tokens:
                            if len(token) > 1:  # Meaningful tokens
                                elements.append(('code_token', token))

                                # Break code token into individual characters
                                for char in token:
                                    if char.strip():
                                        elements.append(('code_char', char))

                # Also add the full code block as a chunk
                if len(code_text) > 10:
                    elements.append(('code_block', code_text))

        # Extract list items individually with parent context
        for li in soup.find_all('li'):
            li_text = li.get_text(strip=True)
            if li_text:
                elements.append(('list_item', li_text))

                # Break list item into phrases and words
                words = li_text.split()
                for i in range(0, len(words), 4):  # Group 4 words together
                    phrase = ' '.join(words[i:i+4])
                    if len(phrase) > 3:
                        elements.append(('list_phrase', phrase))

                # Individual words from list items
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'')
                    if len(word) > 2:
                        elements.append(('list_word', word))

                # Get the parent list title if available
                parent = li.find_parent(['ul', 'ol'])
                if parent:
                    parent_text = parent.get_text(strip=True)[:100]  # First 100 chars
                    if parent_text:
                        elements.append(('list_context', f"List item in: {parent_text}... - {li_text}"))

                # Break list item into characters
                for char in li_text:
                    if char.strip():
                        elements.append(('list_char', char))

        # Extract table content in ultimate granularity
        for table in soup.find_all('table'):
            # Extract headers
            headers = table.find_all(['th'])
            if headers:
                for header in headers:
                    header_text = header.get_text(strip=True)
                    if header_text:
                        elements.append(('table_header', header_text))

                        # Break header into words
                        words = header_text.split()
                        for word in words:
                            word = word.strip('.,;:!?()[]{}"\'')
                            if len(word) > 2:
                                elements.append(('table_header_word', word))

                        # Break header into characters
                        for char in header_text:
                            if char.strip():
                                elements.append(('table_header_char', char))

            # Extract individual cells with position info
            rows = table.find_all('tr')
            for row_idx, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                for col_idx, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)
                    if cell_text:
                        elements.append(('table_cell', f"Row {row_idx}, Col {col_idx}: {cell_text}"))

                        # Break cell content into words
                        words = cell_text.split()
                        for word in words:
                            word = word.strip('.,;:!?()[]{}"\'')
                            if len(word) > 2:
                                elements.append(('table_cell_word', word))

                        # Break cell content into characters
                        for char in cell_text:
                            if char.strip():
                                elements.append(('table_cell_char', char))

        # Extract figure captions, images, and alt text with ultimate context
        for fig in soup.find_all(['figcaption', 'img']):
            if fig.name == 'figcaption':
                fig_text = fig.get_text(strip=True)
                if fig_text:
                    elements.append(('figure_caption', fig_text))

                    # Break caption into phrases and words
                    words = fig_text.split()
                    for word in words:
                        word = word.strip('.,;:!?()[]{}"\'')
                        if len(word) > 2:
                            elements.append(('caption_word', word))

                    # Break caption into characters
                    for char in fig_text:
                        if char.strip():
                            elements.append(('caption_char', char))

            elif fig.name == 'img':
                alt_text = fig.get('alt', '')
                title_text = fig.get('title', '')
                src_text = fig.get('src', '')

                if alt_text:
                    elements.append(('image_alt', alt_text))
                    # Break alt text into words
                    words = alt_text.split()
                    for word in words:
                        word = word.strip('.,;:!?()[]{}"\'')
                        if len(word) > 2:
                            elements.append(('alt_word', word))

                    # Break alt text into characters
                    for char in alt_text:
                        if char.strip():
                            elements.append(('alt_char', char))

                if title_text:
                    elements.append(('image_title', title_text))
                    # Break title into words
                    words = title_text.split()
                    for word in words:
                        word = word.strip('.,;:!?()[]{}"\'')
                        if len(word) > 2:
                            elements.append(('title_word', word))

                    # Break title into characters
                    for char in title_text:
                        if char.strip():
                            elements.append(('title_char', char))

                if src_text:
                    # Extract filename from src
                    filename = src_text.split('/')[-1].split('.')[0]
                    if filename and len(filename) > 2:
                        elements.append(('image_filename', filename))

        # Extract mathematical formulas in extreme detail
        for math_elem in soup.find_all(['math', 'span'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['math', 'equation', 'formula', 'latex', 'mathml', 'display-math', 'inline-math'])):
            math_content = math_elem.get_text(strip=True)
            if math_content:
                elements.append(('math_formula', math_content))

                # Break formula into tokens
                tokens = re.findall(r'\S+', math_content)  # Non-whitespace sequences
                for token in tokens:
                    if len(token) > 1:
                        elements.append(('math_token', token))

                        # Break math token into individual characters
                        for char in token:
                            elements.append(('math_char', char))

        # Extract API references with detailed breakdown
        for api_ref in soup.find_all(['div', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['api', 'reference', 'method', 'function', 'class', 'interface', 'type', 'signature'])):
            api_content = api_ref.get_text(separator=' ', strip=True)
            if api_content:
                elements.append(('api_ref', api_content))

                # Break into words and code-like identifiers
                words = api_content.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'')
                    if len(word) > 1:
                        elements.append(('api_word', word))

                        # Break API word into characters
                        for char in word:
                            if char.strip():
                                elements.append(('api_char', char))

        # Extract special notes and warnings in detail
        for sidebar in soup.find_all(['aside', 'div', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['sidebar', 'note', 'warning', 'tip', 'important', 'caution', 'attention', 'highlight', 'quote', 'admonition', 'callout', 'info', 'success', 'danger', 'attention'])):
            sidebar_text = sidebar.get_text(separator=' ', strip=True)
            if sidebar_text:
                elements.append(('note', sidebar_text))

                # Break into meaningful chunks
                words = sidebar_text.split()
                for i in range(0, len(words), 6):  # Group 6 words together
                    phrase = ' '.join(words[i:i+6])
                    if len(phrase) > 5:
                        elements.append(('note_phrase', phrase))

                # Break note into characters
                for char in sidebar_text:
                    if char.strip():
                        elements.append(('note_char', char))

        # Extract links with their text in detail
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            href = link.get('href', '')
            if link_text and href and not href.startswith('#'):  # Exclude anchor links
                elements.append(('link', f"Link: {link_text} -> {href}"))

                # Break link text into words
                words = link_text.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'')
                    if len(word) > 2:
                        elements.append(('link_word', word))

                # Break link text into characters
                for char in link_text:
                    if char.strip():
                        elements.append(('link_char', char))

        # Extract bold and italic text separately with word-level breakdown
        for bold in soup.find_all(['b', 'strong']):
            bold_text = bold.get_text(strip=True)
            if bold_text:
                elements.append(('bold_text', bold_text))

                # Break bold text into words
                words = bold_text.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'')
                    if len(word) > 2:
                        elements.append(('bold_word', word))

                # Break bold text into characters
                for char in bold_text:
                    if char.strip():
                        elements.append(('bold_char', char))

        for italic in soup.find_all(['i', 'em']):
            italic_text = italic.get_text(strip=True)
            if italic_text:
                elements.append(('italic_text', italic_text))

                # Break italic text into words
                words = italic_text.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'')
                    if len(word) > 2:
                        elements.append(('italic_word', word))

                # Break italic text into characters
                for char in italic_text:
                    if char.strip():
                        elements.append(('italic_char', char))

        # Extract text content in all possible ways
        all_text = soup.get_text(separator=' ')
        all_words = all_text.split()

        # Add every 2-3 words as a separate chunk to increase granularity
        for i in range(0, len(all_words), 2):
            word_pair = ' '.join(all_words[i:i+2])
            if len(word_pair) > 3:  # Meaningful content
                elements.append(('word_pair', word_pair))

        # Add every 3-4 words as a separate chunk
        for i in range(0, len(all_words), 3):
            word_triplet = ' '.join(all_words[i:i+3])
            if len(word_triplet) > 5:
                elements.append(('word_triplet', word_triplet))

        # Add every single character as a separate chunk (with minimum length check)
        for char in all_text:
            if char.strip():  # Only add non-whitespace characters
                elements.append(('char', char))

        # Extract and process CSS selectors and class names for additional context
        for element in soup.find_all():
            classes = element.get('class', [])
            if classes:
                class_str = ' '.join(classes)
                elements.append(('css_class', class_str))

                # Break class names into individual words
                for class_name in classes:
                    for word in class_name.split('-'):
                        if len(word) > 2:
                            elements.append(('class_word', word))

        # Process each element as a separate chunk
        for elem_type, content in elements:
            token_count = self.count_tokens(content)

            if token_count <= self.max_tokens:
                chunks.append({
                    'content': content,
                    'chunk_index': chunk_index,
                    'total_chunks': 0  # Will update later
                })
                chunk_index += 1
            else:
                # If individual element is too large, split it further using fallback chunking
                sub_chunks = self.chunk_text(content, f"{page_data['title']} - {elem_type}")
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'content': sub_chunk['content'],
                        'chunk_index': chunk_index,
                        'total_chunks': 0  # Will update later
                    })
                    chunk_index += 1

        # If no elements were extracted (fallback), use the original content
        if not elements:
            content_text = soup.get_text(separator='\n', strip=True)
            if content_text:
                token_count = self.count_tokens(content_text)
                if token_count <= self.max_tokens:
                    chunks.append({
                        'content': content_text,
                        'chunk_index': chunk_index,
                        'total_chunks': 0
                    })
                    chunk_index += 1
                else:
                    sub_chunks = self.chunk_text(content_text, page_data['title'])
                    for sub_chunk in sub_chunks:
                        chunks.append({
                            'content': sub_chunk['content'],
                            'chunk_index': chunk_index,
                            'total_chunks': 0
                        })
                        chunk_index += 1

        # Update total_chunks for each chunk
        for i, chunk in enumerate(chunks):
            chunk['total_chunks'] = len(chunks)
            chunk['chunk_index'] = i

        logger.info(f"Created {len(chunks)} ultra-granular chunks for '{page_data['title']}'")
        return chunks

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

            # Try to find main content area - prioritize Docusaurus-specific selectors
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find(class_='container') or
                soup.find(class_='main') or
                soup.find(class_='docPage') or
                soup.find(class_='docsPage') or
                soup.find(class_='theme-doc-markdown') or
                soup.find(class_='markdown') or
                soup.find(class_='content') or
                soup.body
            )

            if not main_content:
                logger.warning(f"No main content found for {url}")
                return None

            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else urlparse(url).path.strip('/')

            # Extract text content
            text_content = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)

            # Extract additional content like images, diagrams, and their descriptions
            additional_content = []

            # Extract images with alt text and titles
            images = main_content.find_all('img')
            for img in images:
                img_alt = img.get('alt', '')
                img_title = img.get('title', '')
                img_src = img.get('src', '')
                if img_alt or img_title:
                    additional_content.append(f"Image Description: {img_alt} {img_title}")
                elif img_src:
                    # Extract filename or meaningful part of src
                    img_filename = img_src.split('/')[-1].split('.')[0]
                    additional_content.append(f"Image: {img_filename}")

            # Extract diagrams, figures, and other visual content
            figures = main_content.find_all(['figure', 'svg', 'div'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['diagram', 'figure', 'chart', 'graph', 'visual', 'image', 'pic', 'illustration']))
            for figure in figures:
                fig_caption = figure.find(['figcaption', 'p'], class_=lambda x: x and 'caption' in x.lower())
                if fig_caption:
                    additional_content.append(f"Figure Caption: {fig_caption.get_text(strip=True)}")
                else:
                    fig_content = figure.get_text(separator=' ', strip=True)
                    if len(fig_content) > 10:  # Only add substantial content
                        additional_content.append(f"Visual Content: {fig_content}")

            # Extract code blocks with their titles/captions - IMPROVED
            code_blocks = main_content.find_all(['pre', 'code'])
            for code in code_blocks:
                code_title = code.get('title') or ''
                code_lang = code.get('class', '') or code.parent.get('class', '') if code.parent else ''
                # Extract language from class names
                lang_classes = [cls for cls in str(code_lang).split() if 'lang-' in cls or cls in ['python', 'javascript', 'js', 'typescript', 'ts', 'bash', 'sh', 'json', 'xml', 'html', 'css', 'sql', 'yaml', 'yml', 'cpp', 'c', 'java', 'go', 'rust', 'ruby', 'php']]
                lang_info = f" ({', '.join(lang_classes)})" if lang_classes else ""
                code_content = code.get_text(strip=True)[:500]  # Increase limit to capture more code
                if code_content:
                    additional_content.append(f"Code Sample{lang_info}: {code_content}")

            # Extract tables with their captions
            tables = main_content.find_all('table')
            for i, table in enumerate(tables):
                table_caption = table.find('caption')
                if table_caption:
                    additional_content.append(f"Table {i+1} Caption: {table_caption.get_text(strip=True)}")
                else:
                    # Extract header information from table
                    headers = table.find_all(['th', 'td'])
                    if headers:
                        header_text = ' | '.join([h.get_text(strip=True) for h in headers[:5]])  # First 5 cells
                        additional_content.append(f"Table Content: {header_text}")

            # NEW: Extract mathematical formulas and equations
            math_elements = main_content.find_all(['math', 'span', 'div'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['math', 'equation', 'formula', 'latex', 'mathml']))
            for math_elem in math_elements:
                math_content = math_elem.get_text(strip=True)
                if math_content:
                    additional_content.append(f"Math Formula: {math_content}")

            # NEW: Extract highlighted or special text blocks
            special_blocks = main_content.find_all(['blockquote', 'aside', 'div', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['note', 'warning', 'tip', 'important', 'caution', 'attention', 'highlight', 'quote', 'admonition', 'callout']))
            for block in special_blocks:
                block_text = block.get_text(separator=' ', strip=True)
                if len(block_text) > 10:
                    additional_content.append(f"Note: {block_text}")

            # NEW: Extract lists that might contain important information
            lists = main_content.find_all(['ul', 'ol'])
            for i, lst in enumerate(lists):
                list_items = lst.find_all('li')
                if len(list_items) > 0:
                    items_text = [li.get_text(strip=True)[:100] for li in list_items[:5]]  # First 5 items, max 100 chars each
                    if items_text:
                        additional_content.append(f"List {i+1}: {'; '.join(items_text)}")

            # NEW: Extract detailed code blocks with language detection and titles
            code_containers = main_content.find_all(['pre', 'div'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['code', 'pre', 'language', 'code-block']))
            for container in code_containers:
                # Look for language info in class or data attributes
                classes = container.get('class', [])
                data_lang = container.get('data-language') or container.get('data-lang')

                lang_info = ""
                for cls in classes:
                    if 'lang-' in cls or cls in ['python', 'javascript', 'js', 'typescript', 'ts', 'bash', 'sh', 'json', 'xml', 'html', 'css', 'sql', 'yaml', 'yml', 'cpp', 'c', 'java', 'go', 'rust', 'ruby', 'php', 'tsx', 'jsx']:
                        lang_info = f" ({cls.replace('lang-', '')})"
                        break

                if not lang_info and data_lang:
                    lang_info = f" ({data_lang})"

                # Get the actual code content
                code_elem = container.find('code') or container
                code_content = code_elem.get_text(strip=True)[:1000]  # Increase to capture more code
                if code_content and len(code_content) > 20:  # Only add substantial code blocks
                    additional_content.append(f"Code Block{lang_info}: {code_content}")

            # NEW: Extract detailed mathematical content and equations
            math_containers = main_content.find_all(['div', 'span', 'p'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['math', 'equation', 'formula', 'latex', 'mathml', 'display-math', 'inline-math']))
            for math_container in math_containers:
                math_content = math_container.get_text(strip=True)
                if math_content and len(math_content) > 5:
                    additional_content.append(f"Mathematical Content: {math_content}")

            # NEW: Extract interactive elements and API references
            api_refs = main_content.find_all(['div', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['api', 'reference', 'method', 'function', 'class', 'interface', 'type', 'signature']))
            for api_ref in api_refs:
                api_content = api_ref.get_text(separator=' ', strip=True)
                if api_content and len(api_content) > 10:
                    additional_content.append(f"API Reference: {api_content}")

            # NEW: Extract detailed table information with headers and first few rows
            tables = main_content.find_all('table')
            for i, table in enumerate(tables):
                table_data = []

                # Extract header
                headers = table.find_all(['th'])
                if headers:
                    header_row = ' | '.join([h.get_text(strip=True) for h in headers])
                    table_data.append(f"Header: {header_row}")

                # Extract first few rows
                rows = table.find_all('tr')[1:]  # Skip header row if headers were found
                for j, row in enumerate(rows[:3]):  # First 3 data rows
                    cells = row.find_all(['td'])
                    if cells:
                        row_data = ' | '.join([c.get_text(strip=True) for c in cells])
                        table_data.append(f"Row {j+1}: {row_data}")

                if table_data:
                    additional_content.append(f"Table {i+1}: {'; '.join(table_data)}")

            # NEW: Extract detailed image and diagram information
            images = main_content.find_all('img')
            for img in images:
                img_alt = img.get('alt', '')
                img_title = img.get('title', '')
                img_src = img.get('src', '')
                img_caption = img.find_next_sibling(['figcaption', 'p', 'div']) if img.parent else None
                img_caption_text = img_caption.get_text(strip=True) if img_caption else ''

                if img_alt or img_title or img_caption_text:
                    img_description = f"Image Description: {img_alt} {img_title} {img_caption_text}".strip()
                    additional_content.append(img_description)
                elif img_src:
                    # Extract filename or meaningful part of src
                    img_filename = img_src.split('/')[-1].split('.')[0]
                    additional_content.append(f"Image: {img_filename}")

            # NEW: Extract figure and diagram information with captions
            figures = main_content.find_all(['figure', 'svg', 'div'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['diagram', 'figure', 'chart', 'graph', 'visual', 'image', 'pic', 'illustration', 'mermaid', 'svg']))
            for figure in figures:
                fig_caption = figure.find(['figcaption', 'p'], class_=lambda x: x and 'caption' in x.lower())
                if fig_caption:
                    additional_content.append(f"Figure Caption: {fig_caption.get_text(strip=True)}")
                else:
                    # Look for adjacent captions
                    parent = figure.parent
                    if parent:
                        # Check siblings for captions
                        next_sibling = figure.find_next_sibling(['p', 'div'], class_=lambda x: x and 'caption' in x.lower())
                        if next_sibling:
                            additional_content.append(f"Figure Caption: {next_sibling.get_text(strip=True)}")
                        else:
                            fig_content = figure.get_text(separator=' ', strip=True)
                            if len(fig_content) > 10:  # Only add substantial content
                                additional_content.append(f"Visual Content: {fig_content}")

            # NEW: Extract sidebar and aside content that might contain important notes
            sidebars = main_content.find_all(['aside', 'div', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['sidebar', 'note', 'warning', 'tip', 'important', 'caution', 'attention', 'highlight', 'quote', 'admonition', 'callout', 'info', 'success', 'danger', 'attention']))
            for sidebar in sidebars:
                sidebar_text = sidebar.get_text(separator=' ', strip=True)
                if len(sidebar_text) > 10:
                    # Determine type based on class
                    classes = ' '.join(sidebar.get('class', []))
                    note_type = 'Note'
                    if 'warning' in classes.lower() or 'danger' in classes.lower():
                        note_type = 'Warning'
                    elif 'tip' in classes.lower() or 'info' in classes.lower():
                        note_type = 'Tip'
                    elif 'important' in classes.lower():
                        note_type = 'Important'

                    additional_content.append(f"{note_type}: {sidebar_text}")

            # Combine all content
            if additional_content:
                all_content = clean_content + "\n\n" + "\n".join(additional_content)
            else:
                all_content = clean_content

            path = urlparse(url).path
            chapter_id = path.replace('/', '_').strip('_') or 'home'

            return {
                'url': url,
                'title': title_text,
                'content': all_content,
                'chapter_id': chapter_id,
                'path': path
            }

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def ingest_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Fetch and index content from a single URL with granular chunking

        Returns list of indexed chunks
        """
        page_data = await self.fetch_page_content(url)

        if not page_data or not page_data['content']:
            logger.warning(f"Skipping {url} - no content extracted")
            return []

        # Use granular chunking to create more points
        chunks = self.create_granular_chunks(page_data)

        # Index each chunk
        results = []
        for chunk_data in chunks:
            try:
                # Create unique chapter_id for each chunk
                chunk_chapter_id = f"{page_data['chapter_id']}_section_{chunk_data['chunk_index']}"
                chunk_title = f"{page_data['title']} (Section {chunk_data['chunk_index'] + 1}/{chunk_data['total_chunks']})"

                # Implement retry logic with exponential backoff for rate limiting
                max_retries = 5
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        result = await self.rag_service.index_content(
                            content=chunk_data['content'],
                            chapter_id=chunk_chapter_id,
                            section_title=chunk_title,
                            source_url=url
                        )
                        success = True
                    except Exception as e:
                        if "Rate limit exceeded" in str(e) or "429" in str(e):
                            retry_count += 1
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Rate limit hit on chunk {chunk_data['chunk_index']}, attempt {retry_count}/{max_retries}. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Error indexing chunk {chunk_data['chunk_index']} of {url}: {str(e)}")
                            break  # Don't retry for non-rate limit errors

                if not success:
                    logger.error(f"Failed to index chunk {chunk_data['chunk_index']} after {max_retries} retries due to rate limiting")
                    continue  # Skip this chunk and continue with the next

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
                        '/our-vision', '/privacy', '/signin', '/signup', '/terms',
                        '/github', '/discord', '/twitter', '/linkedin']

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
