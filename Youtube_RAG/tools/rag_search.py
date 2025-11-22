import json
from langchain.tools import tool

@tool
def perform_rag_search(query: str) -> str:
    """Perform RAG search on the loaded video.
    
    Args:
        query(str): Query given by the user to that is to be used in RAG Search.
    """
    return json.dumps({
        "query": query,
        "status": "search_requested"
    })
