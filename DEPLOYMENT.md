# Deployment Guide

## Vercel Deployment

This project is optimized for deployment on Vercel with the following architecture:

### Architecture

- **Frontend**: Docusaurus static site (deployed from `frontend/`)
- **Backend**: Python serverless functions (deployed from `backend/`)

### Memory Optimization

To prevent Out of Memory (OOM) errors during Vercel deployment:

#### Dependencies Split

**Base Dependencies** (`backend/requirements.txt`):
- Core API dependencies (FastAPI, uvicorn, etc.)
- Lightweight ML libraries
- Database and storage clients
- **Excludes** heavy ML models to reduce memory footprint

**Optional ML Dependencies** (`backend/requirements-ml.txt`):
- `fastembed` - Local embedding model
- `langchain` - LangChain framework
- `sentence-transformers` - Sentence embedding models

#### Graceful Degradation

The backend automatically detects if ML dependencies are available:

```python
# Local embeddings are optional
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
```

**Behavior**:
- On Vercel (serverless): Uses OpenRouter API for embeddings (no local models)
- On local development: Can use local ML models as fallback (install with `pip install -r backend/requirements-ml.txt`)

### Configuration

**vercel.json**:
- Configures separate build for frontend and backend
- Sets memory limits for serverless functions
- Defines routing between static site and API endpoints

### Local Development with ML Features

To enable local embedding fallback:

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

### Production Deployment

For Vercel deployment, only base requirements are installed automatically. The application will:
- Use OpenRouter API for all embedding operations
- Log warnings if local embeddings are attempted but unavailable
- Continue to function normally without local ML models

### Monitoring

Check logs for:
- `"Local embeddings not available"` - Expected on Vercel
- `"Switching to local embeddings"` - Should not occur on Vercel
- `"Cannot switch to local embeddings"` - Rate limit hit, but local fallback unavailable

### Build Configuration

The build process:
1. Installs frontend dependencies (`frontend/package.json`)
2. Builds Docusaurus static site
3. Installs backend base dependencies (`backend/requirements.txt`)
4. Deploys as static site + serverless functions

Total memory footprint is optimized to fit within Vercel's limits.
