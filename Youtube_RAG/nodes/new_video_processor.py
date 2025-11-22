from models.state import AgentState
import json
from services.db_data_saver import save_new_video_to_db

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