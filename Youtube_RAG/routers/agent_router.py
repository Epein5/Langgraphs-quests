from models.state import AgentState
import json
from langgraph.graph import END

def routers(state: AgentState):
    """Route to appropriate node based on current state."""
    messages = state["messages"]
    ai_message = messages[-1] if messages else None

    if ai_message and hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        print("[Router] Routing to tool_handler")
        return "tool_handler"

    if ai_message and hasattr(ai_message, 'content'):
        if not ai_message.content or ai_message.content.strip() == "":
            print("[Router] Routing to END (empty response)")
            return END

    tool_message = None
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], 'name'):
            tool_message = messages[i]
            break

    if not tool_message:
        print("[Router] Routing to END (no tool message)")
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
        print("[Router] Routing to END (parse error)")
        return END

    if tool_name == "youtube_video_data_checker":
        if isinstance(content, dict):
            status = content.get("status")
            if status == "not_found":
                print("[Router] Routing to process_new_video")
                return "process_new_video"
            elif status == "found":
                print("[Router] Routing to process_existing_video")
                return "process_existing_video"

    elif tool_name == "perform_rag_search":
        print("[Router] Routing to handle_rag_search")
        return "handle_rag_search"

    print("[Router] Routing to END (default)")
    return END
