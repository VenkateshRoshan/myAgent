"""
FastAPI Backend for Personal AI Agent
Complete API structure with empty endpoints
"""

# -----------------------------------------------
# Import Libraries
# -----------------------------------------------
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import uuid
from datetime import datetime
import random

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import DatabaseManager
from agents.agent import Agent, AgentState

# -----------------------------------------------
# Initialize Everything Required
# -----------------------------------------------

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="Personal AI Agent API",
    description="Modular API for Personal AI Agent with database integration",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Manager
logger.info("Initializing Database Manager...")
db_manager = DatabaseManager()

# Initialize AI Agent
logger.info("Initializing AI Agent...")
agent = Agent(
    model_name=os.getenv("LLAMA_MODEL_NAME", "llama3.1:8b"),
    temperature=float(os.getenv("LLAMA_TEMPERATURE", 0.7))
)

logger.info("FastAPI backend initialized successfully.")

# -----------------------------------------------
# Pydantic Models for Request/Response
# -----------------------------------------------

class ChatRequest(BaseModel):
    query: str
    user_id: str
    conversation_id: str

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    message_id: str
    used_search: bool
    success: bool

class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    last_updated: str

class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    timestamp: str

# -----------------------------------------------
# 1. Core System Endpoints
# -----------------------------------------------

@app.get("/api/health/", tags=["Core"])
async def health_check():
    """System health check"""
    # TODO: Check database connection, agent status, etc.
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# -----------------------------------------------
# 2. User Operations Endpoints
# -----------------------------------------------

@app.post("/api/chat", tags=["Core"])
async def chat_with_agent(chat_request: ChatRequest):
    """
        Chat with the AI agent
    """
    try:
        agent_response = agent.chat(
            original_query=chat_request.query,
            conversation_id=chat_request.conversation_id,
            user_id=chat_request.user_id
        )

        return {
            "answer": agent_response['Agent_response'],
            "used_search": agent_response.get('used_search', False),
            "conversation_id": chat_request.conversation_id,
            "message_id": agent_response['message_id'],
            "success": agent_response.get('success', True)
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request")

# NOTE: Helper functions for user and conversation validation
def validate_user_access(user_id: str):
    user = db_manager.get_user(user_id)  # Sync call
    if not user:
        raise HTTPException(status_code=404, detail="User not found")  # Sync
    return user

def validate_conversation_access(conv_id: str, user_id: str, user):
    conversation = db_manager.get_conversation(conv_id)  # Sync call
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")  # Sync
    if conversation.user_id != user_id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")  # Sync
    return conversation

@app.get("/api/users/user_id={user_id}/convs", 
         response_model=List[ConversationResponse], 
         tags=["User Operations"])
async def get_user_conversations(user_id: str = Path(...)):
    """List user's conversations"""
    # Step 1: Validate user access
    user = validate_user_access(user_id)
    
    # Step 2: Fetch conversations from database
    conversations = db_manager.list_conversations(user_id)
    if not conversations:
        return []
    
    # Step 3: Format response
    response_data = []
    for conv in conversations:
        response_data.append(ConversationResponse(
            conversation_id=conv['conversation_id'],
            title=conv.get('title', 'Untitled'),
            message_count=0,  # Default since not returned
            last_updated=datetime.now().isoformat()  # Default
        ))
    return response_data

@app.get("/api/users/user_id={user_id}/convs/conv_id={conv_id}/", 
         response_model=List[MessageResponse], 
         tags=["User Operations"])
async def get_conversation_messages(user_id: str = Path(...), conv_id: str = Path(...)):
    """Get specific conversation messages"""
    # Step 1: Validate user and conversation access
    user = validate_user_access(user_id)
    conversation = validate_conversation_access(conv_id, user_id, user)
    
    # Step 2: Fetch messages from database
    messages = db_manager.get_conversation_messages(conv_id)
    if not messages:
        return []
    
    # Step 3: Format response
    response_data = []
    for msg in messages:
        response_data.append(MessageResponse(
            message_id=msg['message_id'],
            role=msg['role'],
            content=msg['content'],
            timestamp=msg['timestamp']
        ))
    return response_data

@app.post("/api/users/user_id={user_id}/convs/conv_id={conv_id}/chat", 
          response_model=ChatResponse, 
          tags=["User Operations"])
async def send_message_to_agent(user_id: str = Path(...), 
                               conv_id: str = Path(...), 
                               chat_request: ChatRequest = None):
    """Send message to AI agent"""
    # Step 1: Validate user and conversation access
    user = validate_user_access(user_id)
    conversation = validate_conversation_access(conv_id, user_id, user)
    
    # Step 2: Call AI Agent with the query
    try:
        agent_response = agent.chat(
            original_query=chat_request.query,
            conversation_id=conv_id,
            user_id=user_id
        )

        # Step 3: Save message to database
        await db_save_message({
            'original_query': chat_request.query,
            'enhanced_query': agent_response.get('enhanced_query', chat_request.query),
            'database_sufficient': agent_response.get('database_sufficient', 'insufficient'),
            'search_results': agent_response.get('search_results', []),
            'final_answer': agent_response['Agent_response'],
            'conversation_id': conv_id,
            'user_id': user_id,
            'success': agent_response.get('success', True),
            'conversation_context': agent_response.get('conversation_context', []),
            'similar_messages': agent_response.get('similar_messages', []),
            'used_search': agent_response.get('used_search', False),
            'embeddings': agent_response.get('embeddings', []),
            'model_metadata': agent_response.get('model_metadata', {}),
        })

        return ChatResponse(
            answer=agent_response['Agent_response'],
            conversation_id=conv_id,
            message_id=agent_response['message_id'],
            used_search=agent_response.get('used_search', False),
            success=agent_response.get('success', True)
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request")

@app.delete("/api/users/user_id={user_id}/convs/delete/conv_id={conv_id}/", 
            tags=["User Operations"])
async def delete_user_conversation(user_id: str = Path(...), conv_id: str = Path(...)):
    """Delete user's conversation"""
    # Step 1: Validate user and conversation access
    user = validate_user_access(user_id)
    conversation = validate_conversation_access(conv_id, user_id, user)
    
    # Step 2: Delete conversation
    try:
        await db_delete_conversation(conv_id)
        return {
            "success": True,
            "message": f"Conversation {conv_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

@app.post("/api/users/user_id={user_id}/convs/add/", 
          tags=["User Operations"])
async def create_new_conversation(user_id: str = Path(...)):
    """Create new conversation (returns conv_id)"""
    # Step 1: Validate user access
    user = validate_user_access(user_id)
    
    # Step 2: Generate new conversation ID
    new_conv_id = str(uuid.uuid4())[:8]
    
    # Step 3: Create conversation in database
    try:
        db_manager.save_conversation({
            "conversation_id": new_conv_id,
            "user_id": user_id,
            "title": "New Conversation"
        })
        
        return {
            "success": True,
            "message": "Conversation created successfully",
            "conversation_id": new_conv_id,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

# -----------------------------------------------
# 3. Database Operations Endpoints (Internal/Shared)
# -----------------------------------------------

@app.get("/api/db/users", tags=["Database Operations"])
async def db_get_users():
    """Get users from database"""
    try:
        users = db_manager.list_users()
        return {
            "success": True,
            "count": len(users),
            "users": users
        }
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get users")

@app.get("/api/db/conversations", tags=["Database Operations"])
async def db_get_conversations(user_id: Optional[str] = Query(None)):
    """Get conversations from database"""
    try:
        conversations = db_manager.list_conversations(user_id)
        return {
            "success": True,
            "count": len(conversations),
            "conversations": conversations,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversations")

@app.get("/api/db/messages", tags=["Database Operations"])
async def db_get_messages(conv_id: Optional[str] = Query(None)):
    """Get messages from database"""
    try:
        if conv_id:
            messages = db_manager.get_conversation_messages(conv_id)
        else:
            messages = db_manager.list_messages()
        
        return {
            "success": True,
            "count": len(messages),
            "messages": messages,
            "conversation_id": conv_id
        }
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get messages")

@app.post("/api/db/save_message", tags=["Database Operations"])
async def db_save_message(message_data: dict):
    """Save message to database"""
    try:
        # Call agent's save_to_database function directly with dict
        agent.save_to_database(message_data)
        
        return {
            "success": True,
            "message": "Message saved successfully",
            "conversation_id": message_data.get("conversation_id"),
            "user_id": message_data.get("user_id")
        }
        
    except Exception as e:
        logger.error(f"Error saving message to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.post("/api/db/create_conversation", tags=["Database Operations"])
async def db_create_conversation(conversation_data: dict):
    """Create conversation in database"""
    try:
        db_manager.save_conversation(conversation_data)
        
        return {
            "success": True,
            "message": "Conversation created successfully",
            "conversation_id": conversation_data.get("conversation_id"),
            "user_id": conversation_data.get("user_id")
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

@app.delete("/api/db/delete_conversation", tags=["Database Operations"])
async def db_delete_conversation(conv_id: str = Query(...)):
    """Delete conversation from database"""
    try:
        db_manager.delete_conversation(conv_id)
        
        return {
            "success": True,
            "message": f"Conversation {conv_id} deleted successfully",
            "conversation_id": conv_id
        }
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

@app.delete("/api/db/delete_user", tags=["Database Operations"])
async def db_delete_user(user_id: str = Query(...)):
    """Delete user from database"""
    try:
        db_manager.delete_user(user_id)
        
        return {
            "success": True,
            "message": f"User {user_id} deleted successfully",
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete user")

@app.get("/api/db/search", tags=["Database Operations"])
async def db_search_messages(query: str = Query(...), 
                            user_id: Optional[str] = Query(None)):
    """RAG search operations"""
    try:
        similar_messages = db_manager.search_similar_messages(query, user_id, limit=5)
        
        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "count": len(similar_messages),
            "similar_messages": similar_messages
        }
    except Exception as e:
        logger.error(f"Error searching messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search messages")

@app.post("/api/db/create_user", tags=["Database Operations"])
async def db_create_user(user_data: UserCreate):
    """Create user in database"""
    try:
        db_manager.save_user(user_data.dict())
        
        return {
            "success": True,
            "message": f"User '{user_data.username}' created successfully",
            "username": user_data.username
        }
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create user")

# -----------------------------------------------
# 4. Admin Only Operations
# -----------------------------------------------

@app.get("/api/admin/users", tags=["Admin Operations"])
async def admin_list_users():
    """List all users (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Fetch users from database
    try:
        users = db_manager.list_users()

        response_data = []
        for user in users:
            response_data.append({
                "user_id": user['user_id'],
                "username": user['username'],
                "email": user.get('email', 'N/A'),
                "created_at": user.get('created_at', datetime.now().isoformat()),
                "is_admin": user.get('is_admin', False)
            })
        return response_data
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")

@app.post("/api/admin/users", tags=["Admin Operations"])
async def admin_create_user(user_data: UserCreate):
    """Add new user (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Create new user
    try:
        # Generate unique user_id
        new_user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        user_dict = {
            "user_id": new_user_id,
            "username": user_data.username,
            "email": user_data.email,
            "is_admin": False
        }
        
        db_manager.save_user(user_dict)
        
        return {
            "success": True,
            "message": f"User '{user_data.username}' created successfully",
            "user_id": new_user_id
        }
        
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.delete("/api/admin/users/user_id={user_id}", tags=["Admin Operations"])
async def admin_delete_user(user_id: str = Path(...)):
    """Delete user completely (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Check if user exists
    user = db_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Step 3: Delete user and all related data
    try:
        db_manager.delete_user(user_id)
        
        return {
            "success": True,
            "message": f"User '{user_id}' deleted successfully",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete user")

@app.get("/api/admin/db/view", tags=["Admin Database"])
async def admin_view_database_stats():
    """View database stats & health (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Get database statistics
    try:
        users = db_manager.list_users()
        all_conversations = db_manager.list_conversations()
        all_messages = db_manager.list_messages()
        
        # Calculate stats
        total_users = len(users)
        total_conversations = len(all_conversations)
        total_messages = len(all_messages)
        admin_users = len([u for u in users if u.get('is_admin', False)])
        
        return {
            "database_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_users": total_users,
                "admin_users": admin_users,
                "regular_users": total_users - admin_users,
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "avg_messages_per_conversation": round(total_messages / total_conversations, 2) if total_conversations > 0 else 0
            },
            "recent_activity": {
                "recent_users": users[-5:] if users else [],
                "recent_conversations": all_conversations[-5:] if all_conversations else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get database statistics")

@app.post("/api/admin/db/backup", tags=["Admin Database"])
async def admin_backup_database():
    """Backup database (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Create database backup
    try:
        backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = f"./backups/{backup_filename}"
        
        # Create backup directory if it doesn't exist
        import os
        os.makedirs("./backups", exist_ok=True)
        
        # For SQLite, copy the database file
        import shutil
        shutil.copy2("./aiagent_db.sqlite", backup_path)
        
        return {
            "success": True,
            "message": "Database backup created successfully",
            "backup_filename": backup_filename,
            "backup_path": backup_path,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create database backup")

@app.post("/api/admin/db/cleanup", tags=["Admin Database"])
async def admin_cleanup_database():
    """Clean old data (admin only)"""
    # Step 1: Validate admin permissions
    # TODO: Add proper admin authentication middleware later
    
    # Step 2: Clean old data based on criteria
    try:
        cleanup_stats = {
            "conversations_deleted": 0,
            "messages_deleted": 0,
            "users_affected": 0
        }
        
        # Example cleanup: Delete conversations with no messages
        all_conversations = db_manager.list_conversations()
        
        for conv in all_conversations:
            conv_messages = db_manager.list_messages(conv['conversation_id'])
            if not conv_messages:  # Empty conversation
                db_manager.delete_conversation(conv['conversation_id'])
                cleanup_stats["conversations_deleted"] += 1
        
        return {
            "success": True,
            "message": "Database cleanup completed",
            "cleanup_statistics": cleanup_stats,
            "cleaned_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup database")

# -----------------------------------------------
# Error Handlers & Startup Events
# -----------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "path": str(request.url)}

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    logger.info("API server starting up...")
    # TODO: Initialize database connections
    # TODO: Verify agent is working
    # TODO: Create default admin user if not exists

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API server shutting down...")
    # TODO: Close database connections
    # TODO: Cleanup resources

# -----------------------------------------------
# Main Entry Point
# -----------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,  # Import the FastAPI app from this module
        host="0.0.0.0",
        port=1432,
        reload=False,
        log_level="info"
    )