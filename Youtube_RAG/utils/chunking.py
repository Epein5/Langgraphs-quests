from typing import List
import re


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # First try to split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If no proper sentences found (like in transcripts), split by multiple spaces or newlines
    if len(sentences) == 1:
        # Split on newlines or multiple spaces (common in transcripts)
        sentences = re.split(r'\n+|\s{2,}', text)
    
    return [s.strip() for s in sentences if s.strip()]


def _split_long_sentence(sentence: str, max_size: int) -> List[str]:
    """Split a long sentence into smaller chunks at word boundaries."""
    if len(sentence) <= max_size:
        return [sentence]
    
    words = sentence.split()
    chunks = []
    current = ""
    
    for word in words:
        if len(current) + len(word) + 1 <= max_size:
            current += word + " "
        else:
            if current:
                chunks.append(current.strip())
            current = word + " "
    
    if current:
        chunks.append(current.strip())
    
    return chunks


def _process_sentences(sentences: List[str], chunk_size: int) -> List[str]:
    """Process sentences, splitting any that are too long."""
    processed = []
    for sentence in sentences:
        processed.extend(_split_long_sentence(sentence, chunk_size))
    return processed


def _get_overlap_text(chunk: str, overlap_size: int) -> str:
    """Extract overlap text from end of chunk."""
    return chunk[-overlap_size:] if len(chunk) > overlap_size else chunk


def _add_chunk_if_not_empty(chunks: List[str], chunk: str) -> None:
    """Add chunk to list if it has content."""
    if chunk.strip():
        chunks.append(chunk.strip())


def semantic_chunking(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50
) -> List[str]:
    """
    Chunks text semantically with overlap for better RAG retrieval.
    
    Args:
        text: The input text to chunk (e.g., YouTube transcript)
        chunk_size: Maximum characters per chunk (default: 1000)
        overlap: Number of characters to overlap between chunks (default: 200)
        
    Returns:
        List of text chunks with overlap
    """
    
    if not text:
        return []
    
    # Split into sentences and handle long ones
    sentences = _split_into_sentences(text)
    processed_sentences = _process_sentences(sentences, chunk_size)
    
    # Build chunks with overlap
    chunks = []
    current_chunk = ""
    
    for sentence in processed_sentences:
        will_exceed = len(current_chunk) + len(sentence) + 1 > chunk_size
        
        if not will_exceed:
            current_chunk += sentence + " "
        else:
            _add_chunk_if_not_empty(chunks, current_chunk)
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = overlap_text + sentence + " "
    
    # Add final chunk
    _add_chunk_if_not_empty(chunks, current_chunk)
    
    return chunks
