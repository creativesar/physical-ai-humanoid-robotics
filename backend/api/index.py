import json
from main import app
from mangum import Mangum

# Create the Mangum adapter for serverless deployment
handler = Mangum(app)

def main(event, context):
    # This function will be used by Vercel for serverless deployment
    return handler(event, context)

# For Vercel's Python runtime, we need to handle the request differently
async def handle_request(event, context):
    # Parse the incoming request
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    query_params = event.get("queryStringParameters", {})
    body = event.get("body", "")

    # Create a request object that FastAPI can handle
    # This is a simplified version - in practice, Mangum handles this
    return handler(event, context)