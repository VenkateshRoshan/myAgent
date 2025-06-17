from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import Any, Dict, List, Optional, TypedDict
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
        State that flows b/w all nodes in the graph as a shared memory.
    """
    original_query: str # What the user asked
    enhanced_query: str # Better version of the query
    needs_search: str # Whether the query needs a search
    search_results: str # web search results
    final_answer: str # Final response to the user
    conversation_id: str # To track conversation
    user_id: str # To track user

class myAgent:
    """
        My Personal AI Agent which contains the below functionalities:
        1. Enhance the query
        2. Check if the query needs a search
        3. Search the web if needed
        4. Generate a final answer
    """

    def __init__(self, model_name: str = "llama3", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=self.temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.graph = self.__build_graph__()

    def __build_graph__(self) -> StateGraph:
        """
            LangGraph workflow:
            Query -> Enhance Query -> Check if Search Needed -> Search Web (if needed) -> Generate Final Answer    
        """

        logger.info("Building the state graph for the agent...")
        graph = StateGraph(AgentState)
        graph.add_node(
            "enhance_query",
            self.enhance_query
        )
        graph.add_node(
            "check_search_needed",
            self.check_search_needed,
        )
        graph.add_node(
            "search_web",
            self.search_web,
        )
        graph.add_node(
            "generate_final_answer",
            self.generate_final_answer,
        )

        graph.set_entry_point("enhance_query")

        graph.add_edge(
            "enhance_query", 
            "check_search_needed"
        )
        graph.add_conditional_edges(
            "check_search_needed",
            self._decide_next_step,  # Function to decide next step
            {
                "yes": "search_web",  # If search is needed
                "no": "generate_final_answer"  # If no search is needed
            }
        )

        graph.add_edge(
            "search_web", 
            "generate_final_answer"
        )

        graph.add_edge(
            "generate_final_answer", 
            END
        )

        logger.info("State graph built successfully.")
        return graph.compile()
    
    def enhance_query(self, state: AgentState) -> AgentState:
        """
            Enhance the user's query into a proper prompt for the LLM using the LLM.
        """
        logger.info(f"Enhancing query: ")
        enhanced_prompt = f"""
            You are a prompt engineer. Your job is to take the user's query and if it needs prompting or not enough then you have to rewrite the query 
            for an LLM or else just return the same as it is for this query: {state['original_query']}. If you're rewriting this query then write it as a better and more detailed query.
            You've to return either the original query or just the enhanced query nothing else.
        """
        try:
            messages = [HumanMessage(content=enhanced_prompt)]
            response = self.llm.invoke(messages)

            # Update state with the enhanced query
            state["enhanced_query"] = response.content.strip()
            logger.info(f"✅ Enhanced: '{state['original_query']}' -> '{state['enhanced_query']}'")
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            # Fallback: use original query
            state['enhanced_query'] = state['original_query']

        return state
    
    def check_search_needed(self, state: AgentState) -> AgentState:
        """
            Check if the enhanced query needs a web search.
        """
        logger.info(f"Checking if search is needed for: {state['enhanced_query']}")
        check_prompt = f"""
            You are a prompt engineer. Your job is to take the user's query and check if it needs a web search or not. 
            If it needs a web search then return 'yes' else return 'no' nothing else.
            Query: {state['enhanced_query']}.
            Answer yes if you dont have enough information to answer the query properly so that I can go through the web and search for the more information on the query.
            Answer no if you have enough information to answer the query properly.
        """
        try:
            messages = [HumanMessage(content=check_prompt)]
            response = self.llm.invoke(messages)

            # Update state with the search need
            state["needs_search"] = response.content.strip().lower()
            logger.info(f"✅ Search needed: {state['needs_search']}")
            
        except Exception as e:
            logger.error(f"Error checking search need: {e}")
            # Default to no search needed
            state['needs_search'] = "no"

        return state
    
    def _decide_next_step(self, state: AgentState) -> str:
        """Decides whether to search or answer directly"""
        if state['needs_search'] == "yes":
            return "yes"
        else:
            return "no"
    
    def search_web(self, state: AgentState) -> AgentState:
        """
            Perform a web search using DuckDuckGo if the query needs it.
        """
        logger.info(f"Searching the web for: {state['enhanced_query']}")
        try:
            search_results = self.search_tool.invoke(state['enhanced_query'])
            state["search_results"] = search_results
            logger.info(f"✅ Search results found: {len(search_results)} results")
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            state["search_results"] = "No results found."

        return state
    
    def generate_final_answer(self, state: AgentState) -> AgentState:
        """
            Generate the final answer using the enhanced query and search results.
        """
        logger.info(f"Generating final answer for: {state['enhanced_query']}")
        final_prompt = f"""
            You are a helpful AI assistant. Based on the user's query: {state['enhanced_query']} and the search results: {state['search_results']}, 
            generate a concise and informative answer. If you don't have enough information, state that clearly.
            And you've to give only whatever the user asked dont give any extra information.
        """
        try:
            messages = [HumanMessage(content=final_prompt)]
            response = self.llm.invoke(messages)

            # Update state with the final answer
            state["final_answer"] = response.content.strip()
            logger.info(f"✅ Final answer generated.")
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            state['final_answer'] = "I couldn't generate a final answer due to an error."

        return state
    
    def __chat__(self, original_query: str, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """
            Main entry point for the agent to process a user's query.
        """
        logger.info(f"Processing query: {original_query} for conversation {conversation_id} and user {user_id}")
        
        # Initialize the state
        initial_state = {
            "original_query": original_query,
            "enhanced_query": "",
            "needs_search": "",
            "search_results": "",
            "final_answer": "",
            "conversation_id": conversation_id,
            "user_id": user_id
        }
        
        try:
            # Run the state graph
            result = self.graph.invoke(initial_state)
            logger.info(f"Response generated: {result['final_answer']}")

        except Exception as e:
            logger.error(f"Error during agent processing: {e}")

        return result['final_answer']

if __name__ == "__main__":
    # Example usage
    agent = myAgent(model_name="llama3", temperature=0.7)
    response = agent.__chat__(
        original_query="What is the capital of France?",
        conversation_id="12345",
        user_id="user_1"
    )
    print(response)