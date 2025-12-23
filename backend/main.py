from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our API routes
from api.chat import router as chat_router
from api.content import router as content_router
from api.translate import router as translate_router

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

# Include API routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(content_router, prefix="/api/content", tags=["content"])
app.include_router(translate_router, prefix="/api/translate", tags=["translate"])

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics Textbook API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )