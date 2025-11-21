from typing import List
import os
from openai import AzureOpenAI


def create_embeddings(chunks: List[str], model: str = None) -> List[List[float]]:
    """
    Create vector embeddings for text chunks using OpenAI's embedding model.
    
    Args:
        chunks: List of text chunks to embed
        model: The embedding deployment name (default: uses OPENAI_AZURE_EMBEDDING_DEPLOYMENT from env)
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    if not chunks:
        return []
    
    if model is None:
        model = os.getenv("OPENAI_AZURE_EMBEDDING_DEPLOYMENT")
        if not model:
            raise ValueError("OPENAI_AZURE_EMBEDDING_DEPLOYMENT environment variable is not set")
    
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
        api_key=os.getenv("OPENAI_AZURE_API_KEY"),
        api_version=os.getenv("OPENAI_AZURE_API_VERSION")
    )
    
    embeddings = []
    
    # Process chunks in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        
        # Extract embeddings from response
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings


def create_single_embedding(text: str, model: str = None) -> List[float]:
    """
    Create a vector embedding for a single text string.
    
    Args:
        text: Text to embed
        model: The embedding deployment name (default: uses OPENAI_AZURE_EMBEDDING_DEPLOYMENT from env)
        
    Returns:
        Embedding vector as a list of floats
    """
    if model is None:
        model = os.getenv("OPENAI_AZURE_EMBEDDING_DEPLOYMENT")
        if not model:
            raise ValueError("OPENAI_AZURE_EMBEDDING_DEPLOYMENT environment variable is not set")
    
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
        api_key=os.getenv("OPENAI_AZURE_API_KEY"),
        api_version=os.getenv("OPENAI_AZURE_API_VERSION")
    )
    
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    
    return response.data[0].embedding
