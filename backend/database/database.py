# -----------------------------------------------
# Step 1: Import required libraries and modules
# -----------------------------------------------
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Text,
    Boolean,
    ForeignKey,
    TIMESTAMP,
    JSON
)

from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    scoped_session
)

from sqlalchemy.engine import URL

from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import uuid
import logging
import os

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

from typing import Dict, Any, List

# -----------------------------------------------
# Step 2: Define DB connection (using SQLAlchemy)
# -----------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///aiagent_db.sqlite")
engine = create_engine(DATABASE_URL, echo=False)

# POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg2://postgres:password@localhost:5432/aiagent_db")
# engine = create_engine(POSTGRES_URL, echo=False)
sessionFactory = sessionmaker(bind=engine)
session = scoped_session(sessionFactory)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

CHROMA_COLLECTION_NAME = "message_embeddings"
chroma_collection = chroma_client.get_or_create_collection(
    CHROMA_COLLECTION_NAME
)

# -----------------------------------------------
# Step 3: Define Base ORM class (DeclarativeBase)
# -----------------------------------------------

Base = declarative_base()

# -----------------------------------------------
# Step 4: Define ORM Models
#   - User
#   - UserMetadata
#   - Conversation
#   - ConversationMetadata
#   - Message
#   - MessageMetadata
# -----------------------------------------------

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, nullable=False)
    email = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)

    # Relationships
    user_metadata = relationship("UserMetadata", uselist=False, back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class UserMetadata(Base):
    __tablename__ = "user_metadata"
    
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    personal_info = Column(JSON, default={})
    preferences = Column(JSON, default={})
    long_term_memory = Column(JSON, default={})
    updated_at = Column(TIMESTAMP, default=datetime.utcnow)

    user = relationship("User", back_populates="user_metadata")


class Conversation(Base):
    __tablename__ = "conversations"
    
    conversation_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    title = Column(Text)
    last_updated = Column(TIMESTAMP, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    conv_metadata = relationship("ConversationMetadata", uselist=False, back_populates="conversation")
    messages = relationship("Message", back_populates="conversation")


class ConversationMetadata(Base):
    __tablename__ = "conversation_metadata"
    
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), primary_key=True)
    summary = Column(Text)
    topics = Column(JSON, default=[])
    overall_sentiment = Column(String, default="neutral")
    model_used = Column(String, default="llama3.1:8b")
    system_prompts = Column(JSON, default={})
    updated_at = Column(TIMESTAMP, default=datetime.utcnow)

    # ✅ Fixed this line:
    conversation = relationship("Conversation", back_populates="conv_metadata")


class Message(Base):
    __tablename__ = "messages"
    
    message_id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"))
    user_id = Column(String, ForeignKey("users.user_id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    msg_metadata = relationship("MessageMetadata", uselist=False, back_populates="message")


class MessageMetadata(Base):
    __tablename__ = "message_metadata"
    
    message_id = Column(String, ForeignKey("messages.message_id"), primary_key=True)
    used_search = Column(Boolean, default=False)
    enhanced_query = Column(Text)
    final_json = Column(JSON, default={})
    rag_chunks = Column(JSON, default=[])
    tokens_used = Column(Integer)
    model_version = Column(String, default="llama3.1:8b")
    embedding_score = Column(Integer)
    error_flag = Column(Boolean, default=False)

    message = relationship("Message", back_populates="msg_metadata")


# -----------------------------------------------
# Step 5: Initialize and create tables
# -----------------------------------------------

def create_tables():
    """
    Create all tables in the database.
    """
    try:
        Base.metadata.create_all(engine)
        logging.info("Database tables created successfully.")
    except SQLAlchemyError as e:
        logging.error(f"Error creating database tables: {e}")
        raise

create_tables()
# -----------------------------------------------
# Step 6: Define the DatabaseManager class
# -----------------------------------------------

class DatabaseManager:
    """
    DatabaseManager class to handle all database operations.
    This includes user management, conversation management, message management,
    and metadata management.
    """

    def __init__(self):
        self.session = session()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_function = lambda x: model.encode(x).tolist()


    def close_session(self):
        """
        Close the current database session.
        """
        self.session.close()

    def commit(self):
        """
        Commit the current transaction.
        """
        try:
            self.session.commit()
        except SQLAlchemyError as e:
            logging.error(f"Error committing transaction: {e}")
            self.session.rollback()
            raise e

    # -----------------------------------------------
    # Step 7: Implement User-related methods
    #   - save_user()
    #   - get_user()
    #   - delete_user()
    # -----------------------------------------------

    def save_user(self, user_data: dict):
        """
        Save a new user to the database.
        :param user_data: Dict with user_id, username, email ( optional )
        """
        try:
            user = self.session.query(User).filter_by(user_id=user_data.get('user_id')).first()
            
            if user:
                user.username = user_data.get('username', user.username)
                user.email = user_data.get('email', user.email)
                user.is_admin = user_data.get('is_admin', user.is_admin)
            else:
                user = User(
                    user_id = user_data.get('user_id', str(uuid.uuid4())),
                    username = user_data['username'],
                    email = user_data.get('email', None)
                    
                )
                self.session.add(user)

            self.commit()
            logging.info(f"User {user.username} saved/updated successfully.")
        except Exception as e:
            logging.error(f"Error saving user: {e}")
            self.session.rollback()

    def get_user(self, user_id: str) -> User:
        """
        Retrieve a user by user_id.
        :param user_id: The ID of the user to retrieve.
        :return: User object or None if not found.
        """
        try:
            return self.session.query(User).filter_by(user_id=user_id).first()
        except SQLAlchemyError as e:
            logging.error(f"Error retrieving user {user_id}: {e}")
            self.session.rollback()
            return None
        
    def delete_user(self, user_id: str):
        """
        Delete a user and cascade all related data manually.
        """
        try:
            # Delete all messages and metadata
            user_messages = self.session.query(Message).filter_by(user_id=user_id).all()
            for message in user_messages:
                self.session.query(MessageMetadata).filter_by(message_id=message.message_id).delete()
            self.session.query(Message).filter_by(user_id=user_id).delete()

            # Delete conversations and metadata
            conversations = self.session.query(Conversation).filter_by(user_id=user_id).all()
            for convo in conversations:
                self.session.query(ConversationMetadata).filter_by(conversation_id=convo.conversation_id).delete()
            self.session.query(Conversation).filter_by(user_id=user_id).delete()

            # Delete user metadata and user
            self.session.query(UserMetadata).filter_by(user_id=user_id).delete()
            self.session.query(User).filter_by(user_id=user_id).delete()

            self.commit()
            logging.info(f"✅ User '{user_id}' and related data deleted.")
        except Exception as e:
            logging.error(f"❌ Failed to delete user '{user_id}': {e}")
            self.session.rollback()

    # -----------------------------------------------
    # Step 8: Implement Conversation-related methods
    #   - save_conversation()
    #   - get_conversation()
    #   - delete_conversation()
    # -----------------------------------------------

    def save_conversation(self, conversation_data: dict):
        """
        Save or update a conversation.
        Expected keys: conversation_id, user_id, title
        """
        try:
            convo = self.session.query(Conversation).filter_by(conversation_id=conversation_data["conversation_id"]).first()

            if convo:
                convo.title = conversation_data.get("title", convo.title)
                convo.last_updated = datetime.utcnow()
            else:
                convo = Conversation(
                    conversation_id=conversation_data["conversation_id"],
                    user_id=conversation_data["user_id"],
                    title=conversation_data.get("title", "Untitled"),
                )
                self.session.add(convo)

            self.commit()
            logging.info(f"✅ Conversation '{convo.conversation_id}' saved/updated.")
        except Exception as e:
            logging.error(f"❌ Failed to save conversation: {e}")
            self.session.rollback()

    def get_conversation(self, conversation_id: str) -> Conversation:
        """
        Fetch a conversation by ID.
        """
        try:
            return self.session.query(Conversation).filter_by(conversation_id=conversation_id).first()
        except Exception as e:
            logging.error(f"❌ Failed to get conversation '{conversation_id}': {e}")
            return None

    def delete_conversation(self, conversation_id: str):
        """
        Delete a conversation and all related messages + metadata.
        """
        try:
            # Delete messages & their metadata
            messages = self.session.query(Message).filter_by(conversation_id=conversation_id).all()
            for msg in messages:
                self.session.query(MessageMetadata).filter_by(message_id=msg.message_id).delete()
            self.session.query(Message).filter_by(conversation_id=conversation_id).delete()

            # Delete conversation metadata
            self.session.query(ConversationMetadata).filter_by(conversation_id=conversation_id).delete()

            # Delete conversation
            self.session.query(Conversation).filter_by(conversation_id=conversation_id).delete()

            self.commit()
            logging.info(f"✅ Conversation '{conversation_id}' and related data deleted.")
        except Exception as e:
            logging.error(f"❌ Failed to delete conversation: {e}")
            self.session.rollback()

    # -----------------------------------------------
    # Step 9: Implement Message-related methods
    #   - save_message()
    #   - get_conversation_messages()
    # -----------------------------------------------

    def save_message(self, message_data: dict):
        """
        Save a user or agent message.
        Expected keys: message_id, conversation_id, user_id, role, content, timestamp, metadata (optional)
        """
        try:
            msg = self.session.query(Message).filter_by(message_id=message_data["message_id"]).first()

            if not msg:
                msg = Message(
                    message_id=message_data["message_id"],
                    conversation_id=message_data["conversation_id"],
                    user_id=message_data["user_id"],
                    role=message_data["role"],  # "user" or "agent"
                    content=message_data["content"],
                    timestamp=message_data.get("timestamp", datetime.utcnow())
                )
                self.session.add(msg)
            else:
                # Optional: allow message updates (content correction)
                msg.content = message_data["content"]
                msg.timestamp = message_data.get("timestamp", msg.timestamp)

            self.commit()
            logging.info(f"✅ Message '{msg.message_id}' saved/updated.")
        except Exception as e:
            logging.error(f"❌ Failed to save message: {e}")
            self.session.rollback()

    def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> list:
        """
        Get the last N messages in a conversation.
        Default limit = 50 messages (sorted by timestamp ascending).
        """
        try:
            messages = (
                self.session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.timestamp.asc())
                .limit(limit)
                .all()
            )
            result = []
            for msg in messages:
                result.append({
                    "message_id": msg.message_id,
                    "user_id": msg.user_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            return result
        except Exception as e:
            logging.error(f"❌ Failed to retrieve messages for '{conversation_id}': {e}")
            return []
        
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        try:
            users = self.session.query(User).all()
            return [{"user_id": user.user_id, "username": user.username, "email": user.email} for user in users]
        except Exception as e:
            logging.error(f"❌ Failed to list users: {e}")
            return []

    def list_conversations(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by user"""
        try:
            query = self.session.query(Conversation)
            if user_id:
                query = query.filter_by(user_id=user_id)
            conversations = query.all()
            return [{"conversation_id": conv.conversation_id, "user_id": conv.user_id, "title": conv.title} for conv in conversations]
        except Exception as e:
            logging.error(f"❌ Failed to list conversations: {e}")
            return []

    def list_messages(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """List messages, optionally filtered by conversation"""
        try:
            query = self.session.query(Message)
            if conversation_id:
                query = query.filter_by(conversation_id=conversation_id)
            messages = query.all()
            return [{"message_id": msg.message_id, "role": msg.role, "content": msg.content} for msg in messages]
        except Exception as e:
            logging.error(f"❌ Failed to list messages: {e}")
            return []

    # -----------------------------------------------
    # Step 10: Implement Metadata-related methods
    #   - save_message_metadata()
    #   - get_message_metadata()
    #   - save_conversation_metadata()
    #   - get_conversation_metadata()
    # -----------------------------------------------

    def save_message_metadata(self, metadata_data: dict):
        """
        Save metadata for a message.
        Expected: message_id, key, value (value can be str, json)
        """
        try:
            meta = MessageMetadata(
                message_id=metadata_data["message_id"],
                key=metadata_data["key"],
                value=metadata_data["value"]
            )
            self.session.add(meta)
            self.commit()
            logging.info(f"✅ Metadata saved for message '{meta.message_id}'")
        except Exception as e:
            logging.error(f"❌ Failed to save message metadata: {e}")
            self.session.rollback()

    def save_conversation_metadata(self, conversation_id: str, key: str, value: str):
        """
        Save metadata for a conversation (e.g., summary, tone, topic).
        """
        try:
            meta = ConversationMetadata(
                conversation_id=conversation_id,
                key=key,
                value=value
            )
            self.session.add(meta)
            self.commit()
            logging.info(f"✅ Metadata saved for conversation '{conversation_id}'")
        except Exception as e:
            logging.error(f"❌ Failed to save conversation metadata: {e}")
            self.session.rollback()

    def save_user_metadata(self, user_id: str, info_type: str, data: dict):
        """
        Save user-level metadata using JSON columns.
        """
        try:
            user_meta = self.session.query(UserMetadata).filter_by(user_id=user_id).first()
            if not user_meta:
                user_meta = UserMetadata(user_id=user_id)
                self.session.add(user_meta)
            
            if info_type == "personal_info":
                user_meta.personal_info = data
            elif info_type == "preferences":
                user_meta.preferences = data
                
            self.commit()
            logging.info(f"✅ Metadata saved for user '{user_id}'")
        except Exception as e:
            logging.error(f"❌ Failed to save user metadata: {e}")
            self.session.rollback()

    # -----------------------------------------------
    # Step 11: (Optional) Implement RAG search stub
    # -----------------------------------------------

    def save_message_embedding(self, message_id: str, text: str, metadata: dict):
        """
        Save vector embeddings to ChromaDB (for RAG).
        Assumes embedding is generated elsewhere.
        """
        try:
            embedding = self.embedding_function(text)  # You should define or inject this function

            chroma_collection.add(
                documents=[text],
                ids=[message_id],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logging.info(f"✅ Saved embedding for message '{message_id}'")
        except Exception as e:
            logging.error(f"❌ Failed to save embedding: {e}")

    def search_similar_messages(self, query: str, user_id: str, limit: int = 3) -> list:
        """
        Perform a vector similarity search over past messages (RAG).
        Restrict to a specific user.
        """
        try:
            query_embedding = self.embedding_function(query)
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"user_id": user_id}
            )
            matches = []
            if results["documents"][0]:
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    distance = results["distances"][0][i] if "distances" in results else 0
                    matches.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity": round(1 - distance, 3)  # Convert distance to similarity
                    })
            return matches
        except Exception as e:
            logging.error(f"❌ Error in RAG search: {e}")
            return []

    # -----------------------------------------------
    # Step 12: (Optional) Visualization / Summary Helpers
    # -----------------------------------------------

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation title, last message, and message count."""
        try:
            conversation = self.session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            if not conversation:
                return {}

            message_count = self.session.query(Message).filter_by(conversation_id=conversation_id).count()

            return {
                "conversation_id": conversation.conversation_id,
                "title": conversation.title,
                # "last_message": conversation.last_message,
                "message_count": message_count,
                "user_id": conversation.user_id
            }
        except Exception as e:
            logging.error(f"❌ Failed to get conversation summary: {e}")
            return {}

    def get_all_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all conversation summaries for a user."""
        try:
            conversations = self.session.query(Conversation).filter_by(user_id=user_id).all()
            result = []
            for conv in conversations:
                result.append(self.get_conversation_summary(conv.conversation_id))
            return result
        except Exception as e:
            logging.error(f"❌ Failed to retrieve user conversations: {e}")
            return []

    def get_full_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get full ordered conversation history (user and agent messages)."""
        try:
            messages = (
                self.session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.timestamp.asc())
                .all()
            )
            return [
                {
                    "message_id": msg.message_id,
                    "sender": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        except Exception as e:
            logging.error(f"❌ Failed to retrieve conversation: {e}")
            return []

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Return a user's profile and metadata for personalization."""
        try:
            user = self.session.query(User).filter_by(user_id=user_id).first()
            meta = self.session.query(UserMetadata).filter_by(user_id=user_id).first()

            return {
                "user_id": user.user_id,
                "username": user.username,
                # "total_messages": user.total_messages,
                "personal_info": meta.personal_info if meta else {},
                "preferences": meta.preferences if meta else {}
            }
        except Exception as e:
            logging.error(f"❌ Failed to get user profile: {e}")
            return {}

# -----------------------------------------------
# Step 13: (Optional) Test the module with __main__
# -----------------------------------------------

if __name__ == "__main__":
    print("🧪 Running database test...")

    db = DatabaseManager()

    # Create a fake user
    test_user = {
        "user_id": "test_user_123",
        "username": "butchi",
    }
    db.save_user(test_user)

    # Add metadata
    db.save_user_metadata("test_user_123", "personal_info", {"full_name": "Butchi Venkatesh Adari"})
    db.save_user_metadata("test_user_123", "preferences", {"language": "en"})

    # Create conversation
    conversation_id = "test_convo_001"
    db.save_conversation({
        "conversation_id": conversation_id,
        "user_id": "test_user_123",
        "title": "Sample AI discussion"
    })

    # Save messages
    db.save_message({
        "message_id": "msg_user_001",
        "conversation_id": conversation_id,
        "user_id": "test_user_123",
        "role": "user",
        "content": "Hello Agent!",
        "timestamp": datetime.now()
    })

    db.save_message({
        "message_id": "msg_agent_001",
        "conversation_id": conversation_id,
        "user_id": "test_user_123",
        "role": "agent",
        "content": "Hello Butchi! How can I assist you today?",
        "timestamp": datetime.now()
    })

    # Test vector embeddings
    print("\n🔬 Testing Vector Embeddings...")
    
    # Test embedding function
    try:
        test_text = "Hello, this is a test message"
        embedding = db.embedding_function(test_text)
        print(f"✅ Embedding function works! Vector size: {len(embedding)}")
    except Exception as e:
        print(f"❌ Embedding function failed: {e}")

    # Save some test messages with embeddings
    test_messages = [
        {"id": "embed_msg_1", "text": "I love machine learning and AI", "user_id": "test_user_123"},
        {"id": "embed_msg_2", "text": "Python programming is fun", "user_id": "test_user_123"},
        {"id": "embed_msg_3", "text": "What's the weather like today?", "user_id": "test_user_123"}
    ]
    
    # Save embeddings to ChromaDB
    try:
        for msg in test_messages:
            db.save_message_embedding(msg["id"], msg["text"], {"user_id": msg["user_id"]})
        print("✅ Test embeddings saved to ChromaDB")
    except Exception as e:
        print(f"❌ Failed to save embeddings: {e}")

    # Test vector search
    try:
        query = "I enjoy coding in Python"
        results = db.search_similar_messages(query, "test_user_123", limit=2)
        print(f"✅ Vector search works! Query: '{query}'")
        print("   Similar messages found:")
        for i, result in enumerate(results):
            print(f"   {i+1}. '{result['content']}' (similarity: {result['similarity']:.3f})")
    except Exception as e:
        print(f"❌ Vector search failed: {e}")

    # Test full workflow: save message + embedding
    try:
        full_test_msg = {
            "message_id": "full_test_msg",
            "conversation_id": conversation_id,
            "user_id": "test_user_123",
            "role": "user",
            "content": "I'm interested in learning about neural networks and deep learning",
            "timestamp": datetime.now()
        }
        
        # Save to SQL database
        db.save_message(full_test_msg)
        
        # Save to ChromaDB
        db.save_message_embedding(
            full_test_msg["message_id"], 
            full_test_msg["content"], 
            {"user_id": full_test_msg["user_id"], "role": full_test_msg["role"]}
        )
        
        print("✅ Full workflow test passed (SQL + Vector DB)")
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")

    # Read user profile
    print("\n👤 User Profile:")
    print(db.get_user_profile("test_user_123"))

    # Read conversation summary
    print("\n🗂️ Conversation Summary:")
    print(db.get_conversation_summary(conversation_id))

    # Read full conversation
    print("\n📜 Full Conversation:")
    for msg in db.get_full_conversation(conversation_id):
        print(msg)

    # Read all user conversations
    print("\n📚 All Conversations for User:")
    print(db.get_all_user_conversations("test_user_123"))

    print("\n✅ All tests complete!")