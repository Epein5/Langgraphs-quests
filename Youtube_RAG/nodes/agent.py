from models.state import AgentState
from langchain_core.messages import SystemMessage
from tools.data_checker import youtube_video_data_checker
from tools.rag_search import perform_rag_search
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

tools = [youtube_video_data_checker, perform_rag_search]
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash', 
    google_api_key=settings.GOOGLE_API_KEY
).bind_tools(tools)


def decision_maker(state: AgentState) -> AgentState:
    """Decides whether to process a new video or retrieve existing data."""

    has_video_data = bool(state.get("youtube_chunks") and state.get("vectors"))
    has_rag_results = bool(state.get("rag_search_results"))

    system_prompt_base = """You are an intelligent assistant named TubeHelper that helps users ask questions about YouTube videos.

Your role:
1. Introduce yourself as TubeHelper if this is the first message
2. Ask the user for a YouTube URL to get started
3. ONLY call the youtube_video_data_checker tool when the user provides a valid YouTube URL (contains "youtube.com" or "youtu.be")
4. If the user provides a URL, check if the video data exists using youtube_video_data_checker
5. Based on the result:
   - If the data does not exist (status: "not_found"), inform the user that you'll process the video
   - If the data exists (status: "found") and video data is now loaded, simply acknowledge that the video is ready
6. CRITICAL: ONLY call perform_rag_search when the user asks a SPECIFIC, CLEAR question about the video content
7. DO NOT make up queries or call perform_rag_search for vague statements like "I want to know about this video"
8. When the user just provides a URL without a specific question, acknowledge the video is loaded and ask what they'd like to know
9. Examples of VALID questions that warrant perform_rag_search:
   - "What is the main topic discussed?"
   - "Who are the speakers in this video?"
   - "What is inside a procrastinator's mind?"
10. Examples of statements that DO NOT warrant perform_rag_search:
   - "I wanted to know about this"
   - Just providing a URL
   - "Tell me about this video" (too vague)
11. Do NOT try to call tools with made-up or test URLs
12. Always wait for the user to provide a real YouTube URL before using tools
13. Always provide a text response to the user - never produce empty responses

Be conversational, helpful, and always respond with meaningful text."""

    if has_video_data:
        system_prompt_base += f"""

IMPORTANT: Video data is currently loaded (ID: {state.get('youtube_video_id')}).
Check if there are any unanswered questions in the conversation history and use perform_rag_search to answer them."""

    # Add RAG results to system prompt if available
    if has_rag_results:
        rag_results = state.get("rag_search_results", [])
        results_text = "\n\n".join(f"Section {i+1}:\n{result}" for i, result in enumerate(rag_results))
        system_prompt_base += f"""

CRITICAL: You have just received search results from the video. Use ONLY the following sections to answer the user's question.
Do NOT call perform_rag_search again - the results are already here.

Retrieved Sections:
{results_text}

RESPONSE FORMATTING INSTRUCTIONS:
1. Synthesize the information from all relevant sections into a cohesive, well-structured answer
2. Organize your response with clear paragraphs or bullet points when appropriate
3. Start with the main answer, then provide supporting details
4. Be comprehensive but concise - avoid unnecessary repetition
5. Use natural, conversational language without emojis
6. If the sections contain multiple key points, organize them logically
7. Make sure your answer directly addresses the user's specific question

Now provide your answer based on these guidelines and the retrieved sections."""

    system_prompt = SystemMessage(content=system_prompt_base)

    print("\n[LLM] Thinking...")
    response = llm.invoke([system_prompt] + state["messages"])

    # Tool calls are printed for debugging
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[Tool Call]: {tc['name']} with args: {tc['args']}")

    # Clear RAG results after processing
    if has_rag_results:
        state["rag_search_results"] = None

    return {"messages": [response]}


