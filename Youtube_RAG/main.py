from typing import TypedDict, Optional, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.audio_retriver import get_audio_from_youtube
from utils.speech_to_text import audio_to_text
from utils.chunking import semantic_chunking
from utils.embeddings import create_embeddings
from utils.db_handler import store_video_data, retrieve_video_data
from utils.rag_search import semantic_search

load_dotenv()

# ============================================================
# STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]
    youtube_audio: Optional[bytes]
    youtube_transcript: Optional[str]
    youtube_chunks: Optional[list]
    youtube_video_id: Optional[str]
    youtube_url: Optional[str]
    vectors: Optional[list]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def youtube_video_id_retreiver(youtube_url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    import re
    
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
        r'(?:live\/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


def save_new_video_to_db(youtube_video_id: str) -> dict:
    """Process a new YouTube video: retrieve audio, transcribe, chunk, embed, and store in DB."""
    print(f"📥 Downloading audio from YouTube...")
    youtube_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    audio_bytes = get_audio_from_youtube(youtube_url)
    
    print(f"🎤 Transcribing audio to text...")
    transcript = audio_to_text(audio_bytes)
    
    print(f"📝 Creating semantic chunks...")
    chunks = semantic_chunking(transcript)
    
    print(f"🧮 Generating embeddings...")
    vectors = create_embeddings(chunks)
    
    print(f"💾 Storing in database...")
    store_video_data(youtube_video_id, transcript, vectors, summary=None, chunks=chunks)
    
    print(f"✅ Video processed successfully!")
    return {
        "youtube_video_id": youtube_video_id,
        "transcript": transcript,
        "vectors": vectors,
        "chunks": chunks,
        "summary": None,
    }


# ============================================================
# TOOLS
# ============================================================

@tool
def youtube_video_data_checker(youtube_video_url: str) -> dict:
    """Check if video data exists in the database for the given YouTube video ID.
    
    Args:
        youtube_video_url (str): The URL of the YouTube video.
    """
    youtube_video_id = youtube_video_id_retreiver(youtube_video_url)
    video_data = retrieve_video_data(youtube_video_id)
    if video_data is not None:
        return {"status": "found", "video_id": youtube_video_id}
    return {"status": "not_found", "video_id": youtube_video_id}


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


tools = [youtube_video_data_checker, perform_rag_search]
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash', 
    google_api_key=os.getenv("GOOGLE_API_KEY")
).bind_tools(tools)


# ============================================================
# GRAPH NODES
# ============================================================

def decision_maker(state: AgentState) -> AgentState:
    """Decides whether to process a new video or retrieve existing data."""
    
    has_video_data = bool(state.get("youtube_chunks") and state.get("vectors"))
    
    system_prompt_base = """You are an intelligent assistant named TubeHelper that helps users ask questions about YouTube videos.

Your role:
1. Introduce yourself as TubeHelper if this is the first message
2. Ask the user for a YouTube URL to get started
3. ONLY call the youtube_video_data_checker tool when the user provides a valid YouTube URL (contains "youtube.com" or "youtu.be")
4. If the user provides a URL, check if the video data exists using youtube_video_data_checker
5. Based on the result:
   - If the data does not exist (status: "not_found"), inform the user that you'll process the video
   - If the data exists (status: "found") and video data is now loaded, check the conversation history
6. IMPORTANT: After a video is loaded, look back at the conversation history to see if the user asked any questions
7. If you find an unanswered question about the video, immediately call perform_rag_search with that question
8. When calling perform_rag_search, extract the exact question from the user's message
9. Do NOT try to call tools with made-up or test URLs
10. Always wait for the user to provide a real YouTube URL before using tools
11. Always provide a text response to the user - never produce empty responses

Be conversational, helpful, and always respond with meaningful text."""
    
    if has_video_data:
        system_prompt_base += f"""

IMPORTANT: Video data is currently loaded (ID: {state.get('youtube_video_id')}). 
Check if there are any unanswered questions in the conversation history and use perform_rag_search to answer them."""
    
    system_prompt = SystemMessage(content=system_prompt_base)
    
    print("\n🤖 [LLM] Thinking...")
    response = llm.invoke([system_prompt] + state["messages"])
    
    if response.content:
        print(f"💬 [LLM Response]: {response.content}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"🔧 [Tool Call]: {tc['name']} with args: {tc['args']}")
    
    return {"messages": [response]}


def update_state_only(state: AgentState) -> AgentState:
    """Load existing video data from database."""
    print("\n📂 [State Update] Loading existing video data...")
    messages = state["messages"]
    
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name') and messages[i].name == 'youtube_video_data_checker':
            try:
                content = json.loads(messages[i].content)
                video_id = content.get("video_id")
                if video_id:
                    state["youtube_video_id"] = video_id
                    print(f"   Video ID: {video_id}")
                    break
            except (json.JSONDecodeError, TypeError) as e:
                print(f"   ❌ Error parsing tool message: {e}")
    
    youtube_video_id = state.get("youtube_video_id")
    if youtube_video_id:
        response = retrieve_video_data(youtube_video_id)
        if response:
            state["youtube_transcript"] = response["full_transcription"]
            state["youtube_chunks"] = response["chunks"]
            state["vectors"] = response["vectors"]
            print(f"   ✅ Loaded {len(state['youtube_chunks'])} chunks and {len(state['vectors'])} vectors")
    
    return state


def process_new_video_and_update_state(state: AgentState) -> AgentState:
    """Process a new video and update state."""
    print("\n🎬 [New Video] Processing video from scratch...")
    messages = state["messages"]
    
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name') and messages[i].name == 'youtube_video_data_checker':
            try:
                content = json.loads(messages[i].content)
                video_id = content.get("video_id")
                if video_id:
                    state["youtube_video_id"] = video_id
                    break
            except (json.JSONDecodeError, TypeError):
                pass
    
    youtube_video_id = state.get("youtube_video_id")
    if youtube_video_id:
        response = save_new_video_to_db(youtube_video_id)
        state["youtube_transcript"] = response["transcript"]
        state["youtube_chunks"] = response["chunks"]
        state["vectors"] = response["vectors"]
    
    return state


def handle_rag_search(state: AgentState) -> AgentState:
    """Perform actual RAG search on the video."""
    print("\n🔍 [RAG Search] Searching video content...")
    messages = state["messages"]
    
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name') and messages[i].name == 'perform_rag_search':
            try:
                tool_result = json.loads(messages[i].content)
                query = tool_result.get("query")
                print(f"   Query: {query}")
                
                if query and state.get("youtube_chunks") and state.get("vectors"):
                    search_results = semantic_search(query, state["vectors"], state["youtube_chunks"])
                    response_text = "Based on the video, here are the most relevant sections:\n\n" + "\n".join(search_results)
                    print(f"   ✅ Found {len(search_results)} relevant sections")
                    
                    ai_response = AIMessage(content=response_text)
                    state["messages"].append(ai_response)
                    print(f"\n💬 [RAG Response]: {response_text}")
                else:
                    print(f"   ❌ Missing data - query: {bool(query)}, chunks: {bool(state.get('youtube_chunks'))}, vectors: {bool(state.get('vectors'))}")
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                print(f"   ❌ Error: {e}")
            break
    
    return state


def routers(state: AgentState):
    """Route to appropriate node based on current state."""
    messages = state["messages"]
    ai_message = messages[-1] if messages else None
    
    if ai_message and hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        print("➡️  [Router] Routing to tool_handler")
        return "tool_handler"
    
    if ai_message and hasattr(ai_message, 'content'):
        if not ai_message.content or ai_message.content.strip() == "":
            print("➡️  [Router] Routing to END (empty response)")
            return END
    
    tool_message = None
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name'):
            tool_message = messages[i]
            break
    
    if not tool_message:
        print("➡️  [Router] Routing to END (no tool message)")
        return END
    
    tool_name = tool_message.name
    
    try:
        if isinstance(tool_message.content, str):
            try:
                content = json.loads(tool_message.content)
            except json.JSONDecodeError:
                content = tool_message.content
        else:
            content = tool_message.content
    except (TypeError, AttributeError):
        print("➡️  [Router] Routing to END (parse error)")
        return END
    
    if tool_name == "youtube_video_data_checker":
        if isinstance(content, dict):
            status = content.get("status")
            if status == "not_found":
                print("➡️  [Router] Routing to process_new_video")
                return "process_new_video"
            elif status == "found":
                print("➡️  [Router] Routing to process_existing_video")
                return "process_existing_video"
    
    elif tool_name == "perform_rag_search":
        print("➡️  [Router] Routing to handle_rag_search")
        return "handle_rag_search"
    
    print("➡️  [Router] Routing to END (default)")
    return END


# ============================================================
# BUILD GRAPH
# ============================================================

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


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

def main():
    """Run the interactive chatbot."""
    print("=" * 60)
    print("🎥 YouTube RAG Assistant")
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
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Thanks for using YouTube RAG Assistant!")
                break
            
            if not user_input:
                continue
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            print("\n" + "─" * 60)
            
            # Invoke the graph with current state
            result = app.invoke(state)
            
            # Update state with results (preserves video data across turns)
            state = result
            
            print("─" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using YouTube RAG Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()