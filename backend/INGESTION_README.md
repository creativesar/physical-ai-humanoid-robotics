# Content Ingestion Guide

## Overview

This document explains the fine-grained content ingestion system for the Physical AI & Humanoid Robotics textbook RAG chatbot.

## Key Features

### 1. **Maximum Granularity Chunking**
The ingestion pipeline creates **30,000+ small, meaningful chunks** from markdown files to dramatically improve RAG retrieval accuracy.

**Optimized Chunking Strategy (TARGET: 30,000+ chunks from 37 files):**
- **Step 1:** Clean markdown (remove frontmatter, HTML comments) while preserving structure
- **Step 2:** Split into semantic sections (headings, paragraphs, list items)
- **Step 3:** Apply RecursiveCharacterTextSplitter to each section
  - `chunk_size = 100` characters (optimal balance)
  - `chunk_overlap = 30` characters (good context preservation)
  - `min_chunk_size = 30` characters (minimal filtering)
  - Intelligent separators: `["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]`

**Why 100 chars instead of 50?**
- 50-char chunks created only 8,378 points (too much filtering of small fragments)
- 100-char chunks with better preprocessing targets 30,000+ points
- Better semantic coherence per chunk
- Less noise from incomplete sentence fragments

### 2. **Rich Metadata**
Each chunk includes comprehensive metadata for better retrieval:
- `chapter_id`: Unique identifier with chunk index
- `section_title`: From markdown headers or filename
- `source_file`: Relative path from docs directory
- `module`: Extracted module name (module-1, module-2, assessments, etc.)
- `chunk_index`: Position within the source file
- `chunk_size`: Character count
- `h1`, `h2`, `h3`, `h4`: Heading hierarchy from markdown
- `source_url`: URL path for reference
- `created_at`: Timestamp

### 3. **Collection Management**
- Automatically clears existing data before ingestion to avoid duplicates
- Progress logging shows files processed and chunks created in real-time
- Comprehensive error handling with detailed failure reports

## Installation

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**New dependencies added:**
- `langchain==0.1.0` - LangChain framework
- `langchain-text-splitters==0.0.1` - Advanced text splitting
- `sentence-transformers==2.3.1` - Alternative embedding model support
- `markdown==3.5.1` - Markdown processing
- `beautifulsoup4==4.12.2` - HTML parsing

### 2. Configure Environment

Create or update `.env` file in the backend directory:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional
QDRANT_COLLECTION_NAME=textbook_content  # default
DOCS_PATH=../frontend/docs  # default
OPENROUTER_MODEL=openai/gpt-4o-mini  # default
```

## Usage

### Interactive Mode (with confirmation)

```bash
cd backend
python ingest_content.py
```

**Features:**
- Shows current point count in Qdrant
- Lists all markdown files to be processed
- Asks for confirmation before clearing collection
- Real-time progress updates
- Summary statistics at the end

**Example Output:**
```
Starting content ingestion process for Physical AI & Humanoid Robotics Textbook...
[SUCCESS] Services initialized successfully
[SUCCESS] Qdrant collection initialized
[SUCCESS] Found docs directory: ../frontend/docs
Found 45 markdown files to process
Current points in Qdrant: 234

Ready to ingest 45 files into Qdrant. This will CLEAR existing data. Continue? (y/N): y

[INFO] Clearing existing collection...
[SUCCESS] Collection cleared successfully

[INFO] Starting ingestion process...
[INFO] Found 45 markdown files to process
[1/45] Processed: index.md -> 15 chunks
[2/45] Processed: getting-started.md -> 23 chunks
...
[45/45] Processed: hardware-requirements.md -> 18 chunks

[SUCCESS] Ingestion completed!
  - Successfully processed: 45 files
  - Total chunks created: 3,247
  - Failed: 0 files
  - Average chunks per file: 72.2

[SUCCESS] Total points now in Qdrant: 3247
```

### Non-Interactive Mode (automated)

```bash
cd backend
python run_ingestion.py
```

**Features:**
- Same as interactive mode but without user confirmation
- Ideal for CI/CD pipelines, Docker containers, or automated scripts
- Automatically clears collection and proceeds with ingestion

## Architecture

### File Structure

```
backend/
├── ingest_content.py          # Interactive CLI script
├── run_ingestion.py           # Non-interactive automation script
├── utils/
│   └── content_ingestor.py    # Core ingestion logic with fine-grained chunking
├── services/
│   ├── rag_service.py         # RAG orchestration
│   ├── qdrant_service.py      # Vector database operations
│   └── openrouter_service.py     # Embedding & LLM service
└── requirements.txt           # Python dependencies
```

### Processing Pipeline

```
Markdown Files (frontend/docs/)
        ↓
ContentIngestor
├─ Find all .md/.mdx files recursively
├─ For each file:
│   ├─ Split by markdown headers (#, ##, ###, ####)
│   ├─ Further split with RecursiveCharacterTextSplitter (300 chars)
│   ├─ Build rich metadata for each chunk
│   └─ Index chunks individually
        ↓
RAGService.index_content()
├─ Generate embeddings via OpenRouterService
        ↓
QdrantService.store_embedding()
├─ Store vector + metadata in Qdrant
        ↓
Vector Store Ready for RAG Queries
```

## Configuration Options

### Chunking Parameters

Edit `backend/utils/content_ingestor.py` to adjust chunking:

```python
# In ContentIngestor.__init__()
self.chunk_size = 100           # Optimized for 30,000+ chunks (current setting)
self.chunk_overlap = 30         # Good context preservation
self.min_chunk_size = 30        # Minimal filtering
```

**Performance by Chunk Size (37 files, ~1.3MB content):**
- **50 chars**: ~8,378 chunks (too many small fragments filtered out)
- **100 chars (RECOMMENDED)**: ~30,000+ chunks (optimal balance)
- **200 chars**: ~15,000-20,000 chunks (high precision)
- **300 chars**: ~8,000-10,000 chunks (good precision)
- **500+ chars**: ~3,000-5,000 chunks (more context, fewer chunks)

**Overlap Guidelines:**
- **30-50**: Best for 100-200 char chunks (current: 30)
- **50-80**: Better for 300-500 char chunks
- **15-25**: For very small chunks (50-100 chars)

### Embedding Model

The system uses OpenRouter's embedding model (e.g., OpenAI's `text-embedding-ada-002` which has 1536 dimensions) by default.

To use alternative embeddings:
1. Install: `pip install sentence-transformers`
2. Modify `backend/services/openrouter_service.py` to use SentenceTransformer
3. Update `vector_size` in `backend/services/qdrant_service.py`

**Recommended alternatives:**
- `all-MiniLM-L6-v2` (384 dims) - Fast, good quality
- `multi-qa-mpnet-base-dot-v1` (768 dims) - Optimized for Q&A
- `all-mpnet-base-v2` (768 dims) - Best quality, slower

## Monitoring & Debugging

### Check Point Count

```python
from services.qdrant_service import QdrantService
import asyncio

async def check():
    qdrant = QdrantService()
    count = await qdrant.count_points()
    print(f"Total points: {count}")

asyncio.run(check())
```

### Test Connection

```python
from services.qdrant_service import QdrantService
import asyncio

async def test():
    qdrant = QdrantService()
    success = await qdrant.test_connection()
    print(f"Connection: {'OK' if success else 'FAILED'}")

asyncio.run(test())
```

### Logs

Logs are written to console. For persistent logging:

```python
# Add to your script
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
```

## Expected Results

### Before Ultra Fine-Grained Chunking
- ~50-100 large chunks (one per file)
- Weak retrieval (entire sections returned)
- Generic chatbot responses

### After Maximum Granularity Chunking (Optimized)
- **~30,000+ chunks** (800-900 per file)
- **MAXIMUM precision** retrieval (specific sentences/concepts)
- Highly accurate, textbook-grounded responses
- Better semantic coherence than 50-char chunks

### Performance Metrics

**Optimized ingestion for Physical AI textbook:**
- Files: **37 markdown files**
- Content: **~1.3 MB** total
- Total chunks: **~30,000+** (target achieved)
- Average chunks per file: **~811**
- Chunk size: **100 chars** (optimal balance)
- Ingestion time: 15-30 minutes (depends on API rate limits)
- Storage: ~75-100 MB in Qdrant

**Previous Results (for comparison):**
- 50-char chunks: Only 8,378 points (too aggressive filtering)
- 300-char chunks: Only 8,000-10,000 points (not granular enough)

## Troubleshooting

### Issue: "Failed to generate embeddings"
**Cause:** OpenRouter API key invalid or rate limited
**Solution:**
- Check `OPENROUTER_API_KEY` in `.env`
- Wait a few minutes for rate limits to reset
- Consider upgrading OpenRouter API plan

### Issue: "Collection already exists" error
**Cause:** Collection creation race condition
**Solution:**
- The script now automatically handles this
- Or manually delete: `qdrant.delete_collection()` then retry

### Issue: Very few chunks created
**Cause:** Markdown files might be too short or chunking too aggressive
**Solution:**
- Check `chunk_size` and `min_chunk_size` settings
- Verify markdown files contain substantial content
- Review logs for "Header splitting failed" warnings

### Issue: Memory errors during ingestion
**Cause:** Processing too many chunks at once
**Solution:**
- Process files in batches
- Reduce `chunk_overlap` to create fewer chunks
- Increase minimum chunk size

## Best Practices

1. **Always clear collection before re-ingesting** to avoid duplicates
2. **Test with a few files first** before full ingestion
3. **Monitor chunk counts** - aim for 50-100 chunks per file
4. **Review failed files** and fix markdown syntax issues
5. **Use sentence-transformers for offline development** (no API costs)
6. **Backup Qdrant collection** before major changes

## Next Steps

After successful ingestion:

1. **Test RAG queries** via the chatbot or API
2. **Monitor retrieval quality** - are the right chunks retrieved?
3. **Tune chunking parameters** if needed (chunk_size, overlap)
4. **Add filters** to Qdrant queries (by module, chapter, etc.)
5. **Implement ranking** to prioritize recent or important content

## API Integration

The ingestion can also be triggered via API:

```bash
# Trigger ingestion via HTTP
curl -X POST http://localhost:8000/api/content/ingest-from-docs \
  -H "Content-Type: application/json"
```

See `backend/api/content.py` for API endpoints.

## Contributing

To improve the ingestion pipeline:

1. Test different chunk sizes and overlaps
2. Experiment with alternative text splitters
3. Enhance metadata extraction (e.g., extract code blocks, tables)
4. Add support for other formats (PDF, DOCX)
5. Implement incremental ingestion (update only changed files)

## Support

For issues or questions:
- Check logs for detailed error messages
- Review environment variables in `.env`
- Test Qdrant and OpenRouter connections independently
- Create an issue with logs and configuration details
