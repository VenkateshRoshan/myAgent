"""
WhatsApp-style Streamlit UI for Personal AI Agent
UPDATED VERSION: Full integration with FastAPI backend, database, and agent
Works with: api.py, database.py, agent.py
"""

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="Personal AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = "http://localhost:1432"  # FastAPI server

# Custom CSS for WhatsApp-style UI
def load_css():
    st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Message bubbles */
    .user-message {
        background: #dcf8c6;
        color: #000000;
        border-radius: 15px 15px 5px 15px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .agent-message {
        background: white;
        color: #000000;
        border-radius: 15px 15px 15px 5px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        max-width: 75%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .message-time {
        font-size: 11px;
        color: #667781;
        margin-top: 5px;
        text-align: right;
    }
    
    /* Header styling */
    .chat-header {
        background: #25d366;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 18px;
    }
    
    .conversation-item {
        background: #f0f2f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e4e6ea;
        transition: all 0.3s ease;
    }
    
    .conversation-item:hover {
        background: #e4e6ea;
        transform: translateY(-2px);
    }
    
    /* Clear floats */
    .clearfix::after {
        content: "";
        display: table;
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables for the app"""
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'  # home, conversations, chat

# ===========================================
# API INTEGRATION FUNCTIONS
# ===========================================

def check_agent_health() -> bool:
    """Check if the FastAPI server and agent are running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health/", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"API connection error: {e}")
        return False

def create_user_via_api(username: str) -> Dict[str, Any]:
    """Create a new user through the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/db/create_user",
            json={"username": username, "email": None},
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_users_from_api() -> List[Dict[str, Any]]:
    """Fetch all users from the database via API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/db/users", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("users", [])
        else:
            st.error(f"Failed to fetch users: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []

def get_user_conversations_from_api(user_id: str) -> List[Dict[str, Any]]:
    """Fetch user's conversations from the database via API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/users/user_id={user_id}/convs",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch conversations: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return []

def create_conversation_via_api(user_id: str) -> Dict[str, Any]:
    """Create a new conversation through the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/users/user_id={user_id}/convs/add/",
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_conversation_messages_from_api(user_id: str, conv_id: str) -> List[Dict[str, Any]]:
    """Fetch conversation messages from the database via API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/users/user_id={user_id}/convs/conv_id={conv_id}/",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch messages: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching messages: {e}")
        return []

def send_message_to_agent_via_api(user_id: str, conv_id: str, message: str) -> Dict[str, Any]:
    """Send message to AI agent through the proper API endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/users/user_id={user_id}/convs/conv_id={conv_id}/chat",
            json={
                "query": message,
                "user_id": user_id,
                "conversation_id": conv_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}: {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Connection error: {str(e)}"
        }

def delete_conversation_via_api(user_id: str, conv_id: str) -> Dict[str, Any]:
    """Delete a conversation through the API"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/users/user_id={user_id}/convs/delete/conv_id={conv_id}/",
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ===========================================
# UI COMPONENTS
# ===========================================

def render_user_list():
    """Render the home page with user list from database"""
    st.markdown('<div class="chat-header">🤖 Personal AI Agent</div>', unsafe_allow_html=True)
    
    # Check agent status
    agent_status = check_agent_health()
    if agent_status:
        st.success("🟢 Agent is online and ready!")
    else:
        st.error("🔴 Agent is offline. Please start your FastAPI server.")
        st.info("Run: `python api.py`")
        return  # Don't show users if API is down
    
    st.markdown("### 👥 Select a user:")
    
    # Fetch users from database via API
    users = get_users_from_api()
    
    if users:
        # Display existing users
        for user in users:
            user_id = user.get('user_id', 'unknown')
            username = user.get('username', 'Unknown User')
            
            if st.button(f"💬 {username}", key=f"user_{user_id}", use_container_width=True):
                st.session_state.current_user = user_id
                st.session_state.page = 'conversations'
                st.rerun()
    else:
        st.info("No users found in database. Create a new user below.")
    
    # Add user section
    st.markdown("---")
    st.markdown("### ➕ Add New User:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_user = st.text_input("Enter username:", placeholder="e.g., john_doe")
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_user:
                # Create user via API
                result = create_user_via_api(new_user.lower())
                if result["success"]:
                    st.success(f"User '{new_user}' created successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to create user: {result['error']}")
            else:
                st.warning("Please enter a username!")

def render_conversations_list():
    """Render conversations list for the selected user from database"""
    user_id = st.session_state.current_user
    
    # Header with back button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⬅️", help="Back to users"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        st.markdown(f'<div class="chat-header">💬 Conversations - {user_id}</div>', 
                   unsafe_allow_html=True)
    
    # Fetch conversations from database via API
    conversations = get_user_conversations_from_api(user_id)
    
    if conversations:
        st.markdown("### 📝 Your Conversations:")
        
        for conv in conversations:
            conv_id = conv['conversation_id']
            title = conv.get('title', 'Untitled')
            message_count = conv.get('message_count', 0)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"💬 {title} ({message_count} messages)", 
                           key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.current_conversation = conv_id
                    st.session_state.page = 'chat'
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation"):
                    # Delete conversation via API
                    result = delete_conversation_via_api(user_id, conv_id)
                    if result["success"]:
                        st.success(f"Conversation deleted!")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete: {result['error']}")
    else:
        st.info("No conversations yet. Create a new one!")
    
    # Create new conversation button
    st.markdown("---")
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        # Create conversation via API
        result = create_conversation_via_api(user_id)
        if result["success"]:
            new_conv_id = result["data"]["conversation_id"]
            st.session_state.current_conversation = new_conv_id
            st.session_state.page = 'chat'
            st.rerun()
        else:
            st.error(f"Failed to create conversation: {result['error']}")

def render_chat_interface():
    """Render WhatsApp-style chat interface with database integration"""
    user_id = st.session_state.current_user
    conv_id = st.session_state.current_conversation
    
    # Header with back button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⬅️", help="Back to conversations"):
            st.session_state.page = 'conversations'
            st.rerun()
    
    with col2:
        st.markdown(f'<div class="chat-header">🤖 Chat - {user_id}</div>', 
                   unsafe_allow_html=True)
    
    # Messages container with dynamic height
    container_height = 400
    
    with st.container(height=container_height):
        # Fetch messages from database via API
        messages = get_conversation_messages_from_api(user_id, conv_id)
        
        if messages:
            for message in messages:
                current_time = datetime.now().strftime("%H:%M")
                
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        {message['content']}
                        <div class="message-time">{current_time}</div>
                    </div>
                    <div class="clearfix"></div>
                    """, unsafe_allow_html=True)
                
                elif message['role'] == 'agent':
                    st.markdown(f"""
                    <div class="agent-message">
                        {message['content']}
                        <div class="message-time">{current_time}</div>
                    </div>
                    <div class="clearfix"></div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px; color: #667781;">
                👋 Start your conversation!
            </div>
            """, unsafe_allow_html=True)
    
    # Fixed input section
    st.markdown("---")
    
    # Input form
    with st.form(key=f"message_form_{conv_id}", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_message = st.text_input(
                "Message",
                placeholder="Ask me anything...",
                label_visibility="collapsed",
                key=f"input_{conv_id}"
            )
        
        with col2:
            send_button = st.form_submit_button("📤", use_container_width=True)
    
    # Handle message sending
    if send_button and user_message.strip():
        # Send message to agent via API
        with st.spinner("🤖 AI is thinking..."):
            result = send_message_to_agent_via_api(user_id, conv_id, user_message.strip())
            
            if result["success"]:
                st.success("Message sent!")
                st.rerun()
            else:
                st.error(f"Failed to send message: {result['error']}")

# ===========================================
# MAIN APP
# ===========================================

def main():
    """Main application function"""
    load_css()
    initialize_session_state()
    
    # Page routing based on session state
    if st.session_state.page == 'home':
        render_user_list()
    elif st.session_state.page == 'conversations':
        render_conversations_list()
    elif st.session_state.page == 'chat':
        render_chat_interface()

if __name__ == "__main__":
    main()