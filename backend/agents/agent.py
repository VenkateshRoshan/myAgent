"""
Enhanced AI Agent with Database Integration
Includes full persistence, RAG capabilities, and memory management
"""

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import Any, Dict, List, Optional, TypedDict
import logging
import os
import uuid
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import our database manager
from database.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State that flows between all nodes in the graph"""
    original_query: str
    enhanced_query: str
    database_sufficient: str
    search_results: str
    final_answer: str
    conversation_id: str
    user_id: str
    success: bool
    # Fields for RAG and memory (used only in answer generation)
    conversation_context: str
    similar_messages: List[Dict[str, Any]]

    used_search: bool
    embeddings: Dict[str, Any]
    model_metadata: Dict[str, Any]

class Agent:
    """
        An AI Agent with Database Integration, web search, RAG, and memory management.
        User knowledge tracking.
        # NOTE: need to add voice modality.
    """
    def __init__(self, model_name: str = "llama3.1:8b", 
                temperature: float = 0.7,
                ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=self.temperature,
            base_url=ollama_base_url
        )
        
        # Initialize tools
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Build the enhanced graph
        self.graph = self._build_graph__()
        
        logger.info("✅ Enhanced Agent initialized with database integration")

    def _build_graph__(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with corrected flow"""
        logger.info("Building enhanced state graph...")
        
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("enhance_query", self.enhance_query)
        graph.add_node("load_context", self.load_context)
        graph.add_node("check_database_sufficiency", self.check_database_sufficiency)
        graph.add_node("search_web", self.search_web)
        graph.add_node("generate_final_answer", self.generate_final_answer)
        # graph.add_node("save_to_database", self.save_to_database)

        # Set entry point
        graph.set_entry_point("load_context")

        # Define edges
        graph.add_edge("load_context", "enhance_query")
        graph.add_edge("enhance_query", "check_database_sufficiency")
        
        graph.add_conditional_edges(
            "check_database_sufficiency",
            self._decide_next_step,
            {
                "sufficient": "generate_final_answer",
                "insufficient": "search_web"
            }
        )
        
        graph.add_edge("search_web", "generate_final_answer")
        # NOTE: Future enhancement: Final check 
        """
            after final answer, if not satisfactory, loop back
            graph.add_conditional_edges(
                "generate_final_answer",
                self._final_check,
                {"good": END,
                "bad": "search_web"}
            )
        """
        # NOTE: even look to apply RLHF for better answers later
        # graph.add_edge("generate_final_answer", "save_to_database")
        # graph.add_edge("save_to_database", END)
        graph.add_edge("generate_final_answer", END)

        logger.info("Enhanced state graph built successfully.")
        return graph.compile()
    
    def enhance_query(self, state: AgentState) -> AgentState:
        """Conservative query enhancement - only fix clear problems"""
        logger.info("🔧 Enhancing query (conservative approach)...")
        
        enhanced_prompt = f"""You are a Query Enhancement Specialist.

                            Your job is to ONLY rewrite the user query if it clearly needs improvement.

                            Carefully examine the following query:

                            "{state['original_query']}"

                            ONLY rewrite the query if there are any of the following issues:
                            - Spelling mistakes
                            - Grammar errors
                            - Unclear or vague wording
                            - Missing key information that makes it incomplete
                            - Structurally broken (e.g. run-on sentence, bad punctuation)
                            - Poorly formatted question or prompt for a model/agent

                            DO NOT rewrite if:
                            - The query is already clear, well-formed, and grammatically correct
                            - The query is a short greeting (e.g. "hi", "hello", "good morning")
                            - The query is a command or prompt that is already acceptable (e.g. "generate a summary", "list top AI tools")

                            🔒 Important Rules:
                            - If the query does NOT need fixing, return it EXACTLY as it is — no changes at all.
                            - If the query needs fixing, return only the corrected version — no extra explanations.
                            - Do NOT add or remove meaning unless clearly required for clarity.

                            Now return ONLY the final version of the query — either unchanged or improved. No explanations or commentary.
                        """

        try:
            messages = [HumanMessage(content=enhanced_prompt)]
            response = self.llm.invoke(messages)
            
            enhanced = response.content.strip()
            
            # Check if the response is significantly different (indicating a rewrite was needed)
            if enhanced.lower() != state['original_query'].lower():
                state["enhanced_query"] = enhanced
                logger.info(f"✅ Query rewritten: '{state['original_query']}' -> '{enhanced}'")
            else:
                state["enhanced_query"] = state['original_query']
                logger.info(f"✅ Query kept as-is: '{state['original_query']}'")
            
        except Exception as e:
            logger.error(f"❌ Error enhancing query: {e}")
            state['enhanced_query'] = state['original_query']

        return state
    
    def load_context(self, state: AgentState) -> AgentState:
        """Load conversation context and similar messages for RAG"""
        logger.info("🔍 Loading conversation context and RAG data...")
        
        try:
            # Get recent conversation history (last 10 messages)
            recent_messages = self.db.get_conversation_messages(state['conversation_id'])
            # recent_messages = recent_messages[-10:]  # Last 10 messages
            
            # Format conversation context
            context_parts = []
            for msg in recent_messages:
                speaker = "You" if msg['role'] == 'user' else "Assistant"
                context_parts.append(f"{speaker}: {msg['content']}")
            
            state['conversation_context'] = "\n".join(context_parts) if context_parts else ""
            
            # Search for similar messages using RAG
            similar_messages = self.db.search_similar_messages(
                query=state['original_query'],
                user_id=state['user_id'],
                limit=3 # NOTE : maintain K in config
            )
            state['similar_messages'] = similar_messages
            
            logger.info(f"✅ Loaded context: {len(recent_messages)} recent messages, {len(similar_messages)} similar messages")
            
        except Exception as e:
            logger.error(f"❌ Error loading context: {e}")
            state['conversation_context'] = ""
            state['similar_messages'] = []
        
        return state
    
    def check_database_sufficiency(self, state: AgentState) -> AgentState:
        """Check if database contains enough information to answer the query"""
        logger.info("🤔 Checking database sufficiency...")
        
        # Prepare context information for assessment
        context_info = ""
        if state['conversation_context']:
            context_info += f"Recent conversation context:\n{state['conversation_context']}\n\n"
        
        if state['similar_messages']:
            context_info += "Relevant past discussions:\n"
            for msg in state['similar_messages']:
                context_info += f"- {msg['content'][:200]}...\n"
            context_info += "\n"
        
        check_prompt = f"""You are an AI system that checks whether the provided context is enough to answer a query.

                            Your task is to evaluate if the following query can be fully and accurately answered using ONLY the information given below.

                            🔹 Query:
                            "{state['enhanced_query']}"

                            🔹 Provided Context:
                            {context_info if context_info else "No relevant conversation history or similar discussions found."}

                            🧠 Decision Rules:
                            - If the query can be answered *completely and accurately* using ONLY the above context, respond with **"sufficient"**.
                            - If the context is missing, unrelated, outdated, or lacks enough detail to answer the query, respond with **"insufficient"**.
                            - If the query refers to current events, news, real-time data, or highly specific technical knowledge that is not in the context, respond with **"insufficient"**.
                            - If the query is general (e.g., “What is a Transformer?”, “Summarize this text”) and does NOT rely on any specific context, respond with **"sufficient"**.

                            ⚠️ Return ONLY one word: either "sufficient" or "insufficient". No explanation, no extra content.
                            """


        try:
            messages = [HumanMessage(content=check_prompt)]
            response = self.llm.invoke(messages)
            
            decision = response.content.strip().lower()
            if 'sufficient' in decision:
                state["database_sufficient"] = "sufficient"
            else:
                state["database_sufficient"] = "insufficient"
                
            logger.info(f"✅ Database sufficiency: {state['database_sufficient']}")
            
        except Exception as e:
            logger.error(f"❌ Error checking database sufficiency: {e}")
            state['database_sufficient'] = "insufficient"  # Default to search if error

        return state
    
    def _decide_next_step(self, state: AgentState) -> str:
        """Decide whether database is sufficient or web search is needed"""
        return state['database_sufficient']

    def search_web(self, state: AgentState) -> AgentState:
        """Perform web search"""
        logger.info("🔍 Searching the web...")
        
        try:
            search_results = self.search_tool.invoke(state['enhanced_query'])
            state["search_results"] = search_results
            logger.info(f"✅ Search completed")
            
        except Exception as e:
            logger.error(f"❌ Error during web search: {e}")
            state["search_results"] = "No search results available."

        return state
    
    def generate_final_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer with all available context"""
        logger.info("💭 Generating final answer with full context...")

        # Build comprehensive prompt using structured context
        prompt_parts = [
            f"""You are an intelligent and helpful AI assistant.

                Your job is to answer the user's query using ALL the available information below. 
                If there is conflicting data, prioritize the most recent or most relevant context.

                🔍 User Query:
                "{state['enhanced_query']}"
                """
        ]

        # Add recent conversation context if available
        if state.get('conversation_context'):
            prompt_parts.append(f"""
                            🗣️ Recent Conversation Context:
                            {state['conversation_context']}
                            """)

        # Add relevant past similar messages if available
        if state.get('similar_messages'):
            prompt_parts.append("\n📌 Relevant Past Discussions:")
            for msg in state['similar_messages'][:2]:
                content_snippet = msg['content'].strip().replace('\n', ' ')
                prompt_parts.append(f"- {content_snippet[:150]}...")

        # Add web search results if applicable
        if state.get('search_results') and state['search_results'] != "No search results available.":
            prompt_parts.append(f"""
                    🌐 Web Search Results:
                    {state['search_results']}
                    """)

        # Final instruction to LLM
        prompt_parts.append("""
                    ✅ Now generate a complete, helpful, and context-aware answer using the above information.
                    - Be concise and accurate.
                    - If context is not relevant, fall back to general knowledge.
                    - Do not repeat the query or list the sources unless helpful.
                    Answer:""")

        final_prompt = "\n".join(prompt_parts)

        try:
            messages = [HumanMessage(content=final_prompt)]
            response = self.llm.invoke(messages)

            state["final_answer"] = response.content.strip()
            state["success"] = True
            logger.info("✅ Final answer generated with full context")

        except Exception as e:
            logger.error(f"❌ Error generating final answer: {e}")
            state['final_answer'] = "I encountered an error while generating a response."
            state["success"] = False

        return state
    
    def chat(self, original_query: str, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """Main chat function with full persistence and RAG"""
        logger.info(f"🚀 Processing enhanced chat: {original_query}")
        
        # Initialize state
        initial_state = {
            "original_query": original_query,
            "enhanced_query": "",
            "database_sufficient": "",
            "search_results": "",
            "final_answer": "",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "success": False,
            "conversation_context": "",
            "similar_messages": []
        }
        
        try:
            # Run the enhanced graph
            result = self.graph.invoke(initial_state)
            
            return {
                "Agent_response": result['final_answer'],
                "success": result['success'],
                "conversation_id": conversation_id,
                "user_id": user_id,
                "message_id": f"{user_id}_{conversation_id}_{int(datetime.now().timestamp())}",
                "used_search": result['database_sufficient'] == "insufficient",
                "context_used": bool(result['conversation_context'] or result['similar_messages'])
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced chat: {e}")
            return {
                "Agent_response": f"I encountered an error: {str(e)}",
                "success": False,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "message_id": f"error_{int(datetime.now().timestamp())}",
                "used_search": False,
                "context_used": False
            }
        
    def save_to_database(self, state: AgentState) -> AgentState:
        """Save the conversation to database and create embeddings"""
        logger.info("💾 Saving to database...")
        
        try:
            # Generate unique message IDs
            user_message_id = f"msg_{uuid.uuid4().hex[:8]}"
            agent_message_id = f"msg_{uuid.uuid4().hex[:8]}"
            
            # Save user message
            self.db.save_message({
                "message_id": user_message_id,
                "conversation_id": state['conversation_id'],
                "user_id": state['user_id'],
                "role": "user",
                "content": state['original_query'],
                "timestamp": datetime.now(),
                "metadata": {}
            })
            
            # Save agent message
            self.db.save_message({
                "message_id": agent_message_id,
                "conversation_id": state['conversation_id'],
                "user_id": state['user_id'],
                "role": "agent",
                "content": state['final_answer'],
                "timestamp": datetime.now(),
                "metadata": {
                    "used_search": state['database_sufficient'] == "insufficient",
                    "enhanced_query": state['enhanced_query'],
                    "database_sufficient": state['database_sufficient'],
                    "success": state['success']
                }
            })
            
            # Create embeddings for RAG
            self.db.save_message_embedding(
                message_id=user_message_id,
                text=state['original_query'],
                metadata={
                    "user_id": state['user_id'],
                    "conversation_id": state['conversation_id'],
                    "role": "user",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.db.save_message_embedding(
                message_id=agent_message_id,
                text=state['final_answer'],
                metadata={
                    "user_id": state['user_id'],
                    "conversation_id": state['conversation_id'],
                    "role": "agent",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Update conversation info
            self.db.save_conversation({
                "conversation_id": state['conversation_id'],
                "user_id": state['user_id'],
                "title": state['original_query'][:50] + "..." if len(state['original_query']) > 50 else state['original_query'],
                "last_message": state['final_answer'][:100] + "..." if len(state['final_answer']) > 100 else state['final_answer']
            })
            
            # Update user activity
            self.db.save_user({
                "user_id": state['user_id'],
                "username": state['user_id'],  # For now, same as user_id
                # "total_messages": len(self.db.list_messages())
            })
            
            logger.info("✅ Successfully saved to database with embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error saving to database: {e}")
        
        return state
    
if __name__ == "__main__":
    # Simple test run
    agent = Agent()
    test_response = agent.chat(
        original_query="What is the capital of France?",
        conversation_id="health_check",
        user_id="health_check_user"
    )
    print(test_response)