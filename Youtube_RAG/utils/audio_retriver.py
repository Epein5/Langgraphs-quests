import yt_dlp
import tempfile
import os


def get_audio_from_youtube(youtube_url: str) -> bytes:
    """Downloads audio from YouTube URL and returns it as bytes."""
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    # Download audio to temporary file then read as bytes
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    ydl_opts['outtmpl'] = temp_path.replace('.mp3', '')
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # Read audio bytes
    with open(temp_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Clean up
    os.unlink(temp_path)
    
    return audio_bytes
