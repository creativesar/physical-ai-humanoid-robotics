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
    """
    Check health status of the API and all services
    """
    health_status = {
        "api": "healthy",
        "services": {
            "mistral": None,
            "qdrant": None
        },
        "messages": []
    }
    
    # Test Mistral connection
    try:
        from api.chat import get_mistral_service
        mistral = get_mistral_service()
        mistral_ok = await mistral.test_connection()
        health_status["services"]["mistral"] = mistral_ok
        if mistral_ok:
            health_status["messages"].append("✓ Mistral AI is connected and responding")
        else:
            health_status["messages"].append("✗ Mistral AI connection test failed")
    except Exception as e:
        health_status["services"]["mistral"] = False
        error_msg = str(e)
        if "MISTRAL_API_KEY" in error_msg:
            health_status["messages"].append("✗ Mistral AI: MISTRAL_API_KEY not set or invalid")
        else:
            health_status["messages"].append(f"✗ Mistral AI error: {error_msg[:100]}")
    
    # Test Qdrant connection
    try:
        from api.chat import get_qdrant_service
        qdrant = get_qdrant_service()
        qdrant_ok = await qdrant.test_connection()
        health_status["services"]["qdrant"] = qdrant_ok
        
        if qdrant_ok:
            try:
                point_count = await qdrant.count_points()
                health_status["messages"].append(f"✓ Qdrant is connected ({point_count} documents indexed)")
            except:
                health_status["messages"].append("✓ Qdrant is connected")
        else:
            health_status["messages"].append("✗ Qdrant connection test failed")
    except Exception as e:
        health_status["services"]["qdrant"] = False
        error_msg = str(e)
        if "QDRANT" in error_msg.upper():
            health_status["messages"].append(f"✗ Qdrant: {error_msg[:100]}")
        else:
            health_status["messages"].append(f"✗ Qdrant error: {error_msg[:100]}")
    
    # Overall status
    if health_status["services"]["mistral"] and health_status["services"]["qdrant"]:
        health_status["status"] = "healthy"
    elif health_status["services"]["mistral"] is False or health_status["services"]["qdrant"] is False:
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unknown"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )