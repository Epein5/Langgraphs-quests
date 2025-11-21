import sqlite3
import json
from typing import List, Dict, Optional
import os


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "youtube_rag.db")

def store_video_data(
    primary_key: str,
    full_transcription: str,
    vectors: List[List[float]],
    summary: Optional[str] = None,
    chunks: Optional[List[str]] = None
) -> bool:
    """
    Store YouTube video data with embeddings in SQLite database.
    
    Args:
        primary_key: Unique identifier (e.g., YouTube video ID or URL)
        full_transcription: Complete transcript text
        vectors: List of embedding vectors for the chunks
        summary: Optional summary of the video
        chunks: Optional list of text chunks corresponding to vectors
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.Connection(DB_PATH)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_videos (
                primary_key TEXT PRIMARY KEY,
                full_transcription TEXT NOT NULL,
                summary TEXT,
                chunks TEXT,
                vectors TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Convert to JSON for storage
        vectors_json = json.dumps(vectors)
        chunks_json = json.dumps(chunks) if chunks else None
        
        # Insert or replace
        cursor.execute("""
            INSERT OR REPLACE INTO youtube_videos 
            (primary_key, full_transcription, summary, chunks, vectors, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (primary_key, full_transcription, summary, chunks_json, vectors_json))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Stored: {primary_key}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def retrieve_video_data(primary_key: str) -> Optional[Dict]:
    """
    Retrieve YouTube video data by primary key.
    
    Args:
        primary_key: Unique identifier for the video
        
    Returns:
        Dict with: primary_key, full_transcription, summary, chunks, vectors, timestamps
    """
    try:
        conn = sqlite3.Connection(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM youtube_videos WHERE primary_key = ?
        """, (primary_key,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print(f"⚠️ Not found: {primary_key}")
            return None
        
        return {
            'primary_key': row['primary_key'],
            'full_transcription': row['full_transcription'],
            'summary': row['summary'],
            'chunks': json.loads(row['chunks']) if row['chunks'] else None,
            'vectors': json.loads(row['vectors']),
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
