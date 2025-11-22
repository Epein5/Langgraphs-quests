from langchain.tools import tool
import json
from utils.video_id_retriever import youtube_video_id_retreiver
from utils.db_handler import retrieve_video_data

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

