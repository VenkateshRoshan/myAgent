"""
WhatsApp-style Streamlit UI for Personal AI Agent
Features: User management, Conversation management, WhatsApp-style chat interface
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
API_BASE_URL = "http://localhost:1432"  # Your FastAPI server

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
    if 'users' not in st.session_state:
        st.session_state.users = ['admin']
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}  # {user: [conversation_ids]}
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}  # {conversation_id: messages}
    
    if 'conversation_titles' not in st.session_state:
        st.session_state.conversation_titles = {}  # {conversation_id: title}
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'  # home, conversations, chat

# API functions
def send_message_to_agent(message: str, conversation_id: str, user_id: str) -> Dict[str, Any]:
    """Send message to the AI agent via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={
                "query": message,
                "conversation_id": conversation_id,
                "user_id": user_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: API returned status {response.status_code}",
                "used_search": False,
                "success": False
            }
    
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Error connecting to agent: {str(e)}",
            "used_search": False,
            "success": False
        }

def check_agent_health() -> bool:
    """Check if the AI agent is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_new_conversation(user: str) -> str:
    """Create a new conversation for a user"""
    conversation_id = str(uuid.uuid4())[:8]  # Short ID
    
    if user not in st.session_state.conversations:
        st.session_state.conversations[user] = []
    
    st.session_state.conversations[user].append(conversation_id)
    st.session_state.chat_history[conversation_id] = []
    st.session_state.conversation_titles[conversation_id] = "New Conversation"
    
    return conversation_id

# UI Components
def render_user_list():
    """Render the home page with user list"""
    st.markdown('<div class="chat-header">🤖 Personal AI Agent</div>', unsafe_allow_html=True)
    
    # Check agent status
    agent_status = check_agent_health()
    if agent_status:
        st.success("🟢 Agent is online and ready!")
    else:
        st.error("🔴 Agent is offline. Please start your FastAPI server.")
        st.info("Run: `python backend/main.py`")
    
    st.markdown("### 👥 Select a user:")
    
    # Display users
    for user in st.session_state.users:
        if st.button(f"💬 {user.title()}", key=f"user_{user}", use_container_width=True):
            st.session_state.current_user = user
            st.session_state.page = 'conversations'
            st.rerun()
    
    # Add user section
    st.markdown("---")
    st.markdown("### ➕ Add New User:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_user = st.text_input("Enter username:", placeholder="e.g., john_doe")
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_user and new_user not in st.session_state.users:
                st.session_state.users.append(new_user.lower())
                st.success(f"User '{new_user}' added!")
                st.rerun()
            elif new_user in st.session_state.users:
                st.warning("User already exists!")
            else:
                st.warning("Please enter a username!")

def render_conversations_list():
    """Render conversations list for the selected user"""
    user = st.session_state.current_user
    
    # Header with back button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⬅️", help="Back to users"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        st.markdown(f'<div class="chat-header">💬 Conversations - {user.title()}</div>', 
                   unsafe_allow_html=True)
    
    # Get user conversations
    user_conversations = st.session_state.conversations.get(user, [])
    
    if user_conversations:
        st.markdown("### 📝 Your Conversations:")
        
        for conv_id in user_conversations:
            title = st.session_state.conversation_titles.get(conv_id, "Untitled")
            messages_count = len(st.session_state.chat_history.get(conv_id, []))
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"💬 {title} ({messages_count} messages)", 
                           key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.current_conversation = conv_id
                    st.session_state.page = 'chat'
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation"):
                    st.session_state.conversations[user].remove(conv_id)
                    if conv_id in st.session_state.chat_history:
                        del st.session_state.chat_history[conv_id]
                    if conv_id in st.session_state.conversation_titles:
                        del st.session_state.conversation_titles[conv_id]
                    st.rerun()
    else:
        st.info("No conversations yet. Create a new one!")
    
    # Create new conversation button
    st.markdown("---")
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        conversation_id = create_new_conversation(user)
        st.session_state.current_conversation = conversation_id
        st.session_state.page = 'chat'
        st.rerun()

def get_dynamic_height():
    """Calculate dynamic height based on screen size"""
    # Use JavaScript to get window height and calculate appropriate container height
    return min(max(400, int(st.session_state.get('window_height', 800) * 0.6)), 600)

def render_chat_interface():
    """Render WhatsApp-style chat interface with RESPONSIVE FIXED input"""
    user = st.session_state.current_user
    conv_id = st.session_state.current_conversation
    
    # Header with back button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⬅️", help="Back to conversations"):
            st.session_state.page = 'conversations'
            st.rerun()
    
    with col2:
        conv_title = st.session_state.conversation_titles.get(conv_id, "Chat")
        st.markdown(f'<div class="chat-header">🤖 {conv_title} - {user.title()}</div>', 
                   unsafe_allow_html=True)
    
    # Dynamic height calculation - responsive to window size
    container_height = get_dynamic_height()
    
    # Messages container with dynamic height
    with st.container(height=container_height):
        messages = st.session_state.chat_history.get(conv_id, [])
        
        if messages:
            for message in messages:
                if message['type'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        {message['content']}
                        <div class="message-time">{message['time']}</div>
                    </div>
                    <div class="clearfix"></div>
                    """, unsafe_allow_html=True)
                
                elif message['type'] == 'agent':
                    st.markdown(f"""
                    <div class="agent-message">
                        {message['content']}
                        <div class="message-time">{message['time']}</div>
                    </div>
                    <div class="clearfix"></div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px; color: #667781;">
                👋 Start your conversation!
            </div>
            """, unsafe_allow_html=True)
    
    # FIXED INPUT - This stays at bottom, outside the scrollable container
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
        current_time = datetime.now().strftime("%H:%M")
        
        # Add user message
        if conv_id not in st.session_state.chat_history:
            st.session_state.chat_history[conv_id] = []
        
        st.session_state.chat_history[conv_id].append({
            'type': 'user',
            'content': user_message.strip(),
            'time': current_time
        })
        
        # Update conversation title if it's the first message
        if st.session_state.conversation_titles[conv_id] == "New Conversation":
            title = user_message.strip()[:30] + "..." if len(user_message) > 30 else user_message.strip()
            st.session_state.conversation_titles[conv_id] = title
        
        # Send to agent and get response
        with st.spinner("🤖 AI is thinking..."):
            response = send_message_to_agent(user_message.strip(), conv_id, user)
            
            st.session_state.chat_history[conv_id].append({
                'type': 'agent',
                'content': response['answer'],
                'time': current_time,
                'used_search': response.get('used_search', False)
            })
        
        st.rerun()

# Main app
def main():
    load_css()
    initialize_session_state()
    
    # Page routing
    if st.session_state.page == 'home':
        render_user_list()
    elif st.session_state.page == 'conversations':
        render_conversations_list()
    elif st.session_state.page == 'chat':
        render_chat_interface()

if __name__ == "__main__":
    main()