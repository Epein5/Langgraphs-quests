from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.data_checker import youtube_video_data_checker
from tools.rag_search import perform_rag_search
from routers.agent_router import routers
from models.state import AgentState
from nodes.agent import decision_maker
from nodes.existing_video_porcessor import update_state_only
from nodes.new_video_processor import process_new_video_and_update_state
from nodes.rag_search import handle_rag_search

load_dotenv()


tools = [youtube_video_data_checker, perform_rag_search]


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("decision_maker", decision_maker)
    graph.add_node("update_state_only", update_state_only)
    graph.add_node("process_new_video_and_update_state", process_new_video_and_update_state)
    graph.add_node("handle_rag_search", handle_rag_search)
    
    tool_node = ToolNode(tools)
    graph.add_node("tool_node", tool_node)
    
    # Set entry point
    graph.set_entry_point("decision_maker")
    
    # Add edges
    graph.add_conditional_edges(
        "decision_maker",
        routers,
        {
            "tool_handler": "tool_node",
            "process_new_video": "process_new_video_and_update_state",
            "process_existing_video": "update_state_only",
            "handle_rag_search": "handle_rag_search",
            END: END,
        }
    )
    
    graph.add_conditional_edges(
        "tool_node",
        routers,
        {
            "tool_handler": "tool_node",
            "process_new_video": "process_new_video_and_update_state",
            "process_existing_video": "update_state_only",
            "handle_rag_search": "handle_rag_search",
            END: "decision_maker",
        }
    )
    
    graph.add_edge("update_state_only", "decision_maker")
    graph.add_edge("process_new_video_and_update_state", "decision_maker")
    graph.add_edge("handle_rag_search", "decision_maker")
    
    return graph.compile()

def main():
    """Run the interactive chatbot."""
    print("=" * 60)
    print("YouTube RAG Assistant")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    app = build_graph()
    
    # Initialize state to maintain conversation and video data across turns
    state = {
        "messages": [],
        "youtube_video_id": None,
        "youtube_transcript": None,
        "youtube_chunks": None,
        "vectors": None,
        "rag_search_results": None,
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Thanks for using YouTube RAG Assistant!")
                break
            
            if not user_input:
                continue
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))

            print("\n" + "─" * 60)

            # Stream the graph execution and capture final state
            streamed_response = False

            for event in app.stream(state, stream_mode="updates"):
                # Get node name and state update
                for node_name, node_state in event.items():
                    # Check if this is the decision_maker node with a message
                    if node_name == "decision_maker" and "messages" in node_state:
                        messages = node_state["messages"]
                        if messages and hasattr(messages[0], 'content') and messages[0].content:
                            if not streamed_response:
                                print("\n[Assistant]: ", end="", flush=True)
                                streamed_response = True
                            # Stream the content
                            for char in messages[0].content:
                                print(char, end="", flush=True)

                    # Update state from final node outputs
                    state.update(node_state)

            if streamed_response:
                print()  # New line after streaming

            print("─" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using YouTube RAG Assistant!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()