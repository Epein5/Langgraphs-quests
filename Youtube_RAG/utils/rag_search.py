from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import create_single_embedding


def semantic_search(
    query: str,
    vectors: List[List[float]],
    chunks: List[str],
    top_k: int = 5
) -> List[str]:
    """
    Perform semantic search to find the most similar chunks to a query.
    
    Args:
        query: The search query string
        vectors: List of embedding vectors (each vector is a list of floats)
        chunks: List of text chunks corresponding to the vectors
        top_k: Number of top similar chunks to return (default: 3)
    
    Returns:
        List of the top_k most similar chunks
    """
    if not vectors or not chunks:
        return []
    
    if len(vectors) != len(chunks):
        raise ValueError("Number of vectors must match number of chunks")
    
    # Generate embedding for the query
    query_embedding = create_single_embedding(query)
    
    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity([query_embedding], vectors)[0]
    
    # Get indices of top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the corresponding chunks
    return [chunks[i] for i in top_indices]
