from utils.audio_retriver import get_audio_from_youtube
from utils.speech_to_text import audio_to_text
from utils.embeddings import create_embeddings
from utils.chunking import semantic_chunking
from utils.db_handler import store_video_data

def save_new_video_to_db(youtube_video_id: str) -> dict:
    """Process a new YouTube video: retrieve audio, transcribe, chunk, embed, and store in DB."""
    print("Downloading audio from YouTube...")
    youtube_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    audio_bytes = get_audio_from_youtube(youtube_url)

    print("Transcribing audio to text...")
    transcript = audio_to_text(audio_bytes)

    print("Creating semantic chunks...")
    chunks = semantic_chunking(transcript)

    print("Generating embeddings...")
    vectors = create_embeddings(chunks)

    print("Storing in database...")
    store_video_data(youtube_video_id, transcript, vectors, summary=None, chunks=chunks)

    print("Video processed successfully!")
    return {
        "youtube_video_id": youtube_video_id,
        "transcript": transcript,
        "vectors": vectors,
        "chunks": chunks,
        "summary": None,
    }
