import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Configuration settings for the YouTube RAG application"""
    
    # Google API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Azure OpenAI API
    OPENAI_AZURE_API_KEY = os.getenv("OPENAI_AZURE_API_KEY")
    OPENAI_AZURE_ENDPOINT = os.getenv("OPENAI_AZURE_ENDPOINT")
    OPENAI_AZURE_API_VERSION = os.getenv("OPENAI_AZURE_API_VERSION")
    OPENAI_AZURE_DEPLOYMENT = os.getenv("OPENAI_AZURE_DEPLOYMENT")
    OPENAI_AZURE_EMBEDDING_DEPLOYMENT = os.getenv("OPENAI_AZURE_EMBEDDING_DEPLOYMENT")
    
    @classmethod
    def validate(cls):
        """Validate that required environment variables are set"""
        required = ['GOOGLE_API_KEY']
        missing = [var for var in required if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Create a singleton instance
settings = Settings()
