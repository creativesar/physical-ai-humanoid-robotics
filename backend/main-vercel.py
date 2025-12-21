from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="Backend API for RAG chatbot and content services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import and set up routes, but handle gracefully if dependencies are missing
try:
    from api.chat import router as chat_router
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
    chat_available = True
except ImportError as e:
    logger.warning(f"Chat router not available: {e}")

    @app.post("/api/chat/query")
    async def chat_unavailable():
        return {"error": "Chat service not available in this deployment", "available": False}

    chat_available = False

try:
    from api.content import router as content_router
    app.include_router(content_router, prefix="/api/content", tags=["content"])
    content_available = True
except ImportError as e:
    logger.warning(f"Content router not available: {e}")

    @app.post("/api/content/index")
    async def content_unavailable():
        return {"error": "Content service not available in this deployment", "available": False}

    content_available = False

try:
    from api.translate import router as translate_router
    app.include_router(translate_router, prefix="/api/translate", tags=["translate"])
    translate_available = True
except ImportError as e:
    logger.warning(f"Translate router not available: {e}")

    @app.post("/api/translate/urdu")
    async def translate_unavailable():
        return {"error": "Translation service not available in this deployment", "available": False}

    translate_available = False

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics Textbook API - Vercel Deployment"}

@app.get("/health")
async def health_check():
    """
    Check health status of the API and all services
    """
    health_status = {
        "api": "healthy",
        "services": {
            "chat": chat_available,
            "content": content_available,
            "translate": translate_available
        },
        "message": "API is running with reduced functionality for Vercel deployment"
    }

    if chat_available and content_available and translate_available:
        health_status["status"] = "fully_operational"
    else:
        health_status["status"] = "operational_with_limitations"

    return health_status

# Add fallback endpoints for basic functionality
@app.get("/docs")
async def docs_redirect():
    return {"message": "API documentation available", "endpoints": ["/", "/health", "/api/chat/query", "/api/content/index", "/api/translate/urdu"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )