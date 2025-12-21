import os
from mangum import Mangum
from main import app  # Import your existing FastAPI app

# Create Mangum handler for Vercel serverless deployment
handler = Mangum(app, lifespan="off")

def handler(event, context):
    """
    This is the entry point for Vercel's serverless functions.
    Vercel looks for a function called 'handler' by default
    """
    return handler(event, context)

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)