from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os

# import my agent
from agents.base_agent import myAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="My Personal AI Agent API",
    description="API for My Personal AI Agent with web search capabilities.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
logger.info("Initializing the agent...")
agent = myAgent(
    model_name=os.getenv("LLAMA_MODEL_NAME", "llama3"),
    temperature=float(os.getenv("LLAMA_TEMPERATURE", 0.7))
)
logger.info("Agent initialized successfully.")

class ChatRequest(BaseModel):
    """
        Data Structure for chat requests.
    """
    query: str
    conversation_id: str = None
    user_id: str = "admin"  # Default user_id for simplicity

    def __init__(self, **data):
        if not data.get('conversation_id'):
            data['conversation_id'] = "default_conversation"
        if not data.get('user_id'):
            data['user_id'] = "admin"
        super().__init__(**data)

class ChatResponse(BaseModel):
    """
        Data Structure for chat responses.
    """
    answer: str
    used_search: bool
    conversation_id: str
    success: bool
    message_id: str

# API Endpoints
@app.get("/")
async def root():
    """
        Root endpoint - basic info about the API.
    """

    return {
        "message": "Personal AI Agent API.",
        "version": "1.0.0",
        "status": "Running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "docs": "/docs",
        }
    }

@app.get("/api/health")
async def health_check():
    """
        Health check endpoint to verify if the API is running.
    """

    try:
        test_response = agent.__chat__(
            original_query="Hello, are you there?",
            conversation_id="health_check",
            user_id="health_check_user"
        )
        if test_response['success']:
            return {"status": "healthy", "message": "Agent is operational."}
        else:
            raise HTTPException(status_code=500, detail="Agent is not operational.")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Agent is not operational.")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
        Chat endpoint to interact with the agent.
        It processes the user's query and returns the agent's response.
    """
    try:
        logger.info(f"Received chat request: {request}")
        response = agent.__chat__(
            original_query=request.query,
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        logger.info(f"Agent response: {response}")

        return ChatResponse(
            answer=response['Agent_response'],
            used_search=response.get('used_search', False),
            conversation_id=response['conversation_id'],
            success=response['success'],
            message_id=response['message_id']
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the request.")
    
@app.get("/api/docs")
async def get_docs():
    """
        Endpoint to retrieve API documentation.
    """
    return {
        "message": "API documentation is available at /docs",
        "docs_url": "/docs"
    }

@app.get("/api/info")
async def get_info():
    """
        Endpoint to retrieve basic information about the API.
    """
    return {
        "name": "My Personal AI Agent API",
        "version": "1.0.0",
        "description": "API for My Personal AI Agent with web search capabilities.",
        "status": "Running"
    }

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting the FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1432,
        reload=True,
        log_level="info"
    )