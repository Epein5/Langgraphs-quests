from models.state import AgentState
import json
from utils.db_handler import retrieve_video_data

def update_state_only(state: AgentState) -> AgentState:
    """Load existing video data from database."""
    print("\n[State Update] Loading existing video data...")
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
                print(f"   Error parsing tool message: {e}")

    youtube_video_id = state.get("youtube_video_id")
    if youtube_video_id:
        response = retrieve_video_data(youtube_video_id)
        if response:
            state["youtube_transcript"] = response["full_transcription"]
            state["youtube_chunks"] = response["chunks"]
            state["vectors"] = response["vectors"]
            print(f"   Loaded {len(state['youtube_chunks'])} chunks and {len(state['vectors'])} vectors")

    return state
