from models.state import AgentState
import json
from utils.rag_search import semantic_search

def handle_rag_search(state: AgentState) -> AgentState:
    """Perform actual RAG search on the video."""
    print("\n[RAG Search] Searching video content...")
    messages = state["messages"]

    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name') and messages[i].name == 'perform_rag_search':
            try:
                tool_result = json.loads(messages[i].content)
                query = tool_result.get("query")
                print(f"   Query: {query}")

                if query and state.get("youtube_chunks") and state.get("vectors"):
                    search_results = semantic_search(query, state["vectors"], state["youtube_chunks"])
                    print(f"   Found {len(search_results)} relevant sections")

                    # Store results in state for agent processing
                    state["rag_search_results"] = search_results
                else:
                    print(f"   Missing data - query: {bool(query)}, chunks: {bool(state.get('youtube_chunks'))}, vectors: {bool(state.get('vectors'))}")
                    state["rag_search_results"] = []
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                print(f"   Error: {e}")
                state["rag_search_results"] = []
            break

    return state
