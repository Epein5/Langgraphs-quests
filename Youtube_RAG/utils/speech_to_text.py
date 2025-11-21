import requests
import os
from dotenv import load_dotenv

load_dotenv()


def audio_to_text(audio_bytes: bytes) -> str:
    """
    Converts audio bytes to text using Azure Whisper API.
    
    Args:
        audio_bytes: Audio data in bytes format
        
    Returns:
        Transcribed text from the audio
    """
    
    # Azure Whisper endpoint
    endpoint = "https://grow-me82mm7z-eastus2.cognitiveservices.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_AZURE_API_KEY")
    
    # Set headers
    headers = {
        "api-key": api_key,
    }
    
    # Prepare the file data
    files = {
        "file": ("audio.mp3", audio_bytes, "audio/mpeg")
    }
    
    # Make the request
    response = requests.post(endpoint, headers=headers, files=files)
    response.raise_for_status()
    
    # Extract and return the transcribed text
    result = response.json()
    return result.get("text", "")
