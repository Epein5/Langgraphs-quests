# YouTube RAG Assistant

An interactive chatbot that helps you ask questions about YouTube videos using RAG (Retrieval-Augmented Generation).

## Features

- 🎥 Download and transcribe YouTube videos
- 💾 Store transcripts in local SQLite database
- 🔍 Semantic search across video content
- 💬 Interactive terminal-based chat interface
- 📝 Detailed logging of all operations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Option 1: Using the run script (Recommended)

```bash
./run.sh
```

### Option 2: Using Python directly

```bash
# Activate virtual environment first
source ../venc/bin/activate

# Run the chatbot
python main.py
```

### Option 3: Direct Python path

```bash
/path/to/venv/bin/python main.py
```

### Example Conversation

```
👤 You: Hey

💬 [LLM Response]: Hello! I'm TubeHelper, your YouTube video assistant. Please provide a YouTube URL...

👤 You: I want to search for this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ What is the video about?

🔧 [Tool Call]: youtube_video_data_checker
📂 [State Update] Loading existing video data...
✅ Loaded 10 chunks and 10 vectors

🔧 [Tool Call]: perform_rag_search
🔍 [RAG Search] Searching video content...
💬 [RAG Response]: Based on the video, here are the most relevant sections:
...
```

## Project Structure

```
Youtube_RAG/
├── main.py                 # Main interactive chatbot
├── pipeline.ipynb          # Jupyter notebook for development/testing
├── db/                     # SQLite database storage
└── utils/
    ├── audio_retriver.py   # YouTube audio download
    ├── speech_to_text.py   # Audio transcription
    ├── chunking.py         # Text chunking
    ├── embeddings.py       # Vector embeddings
    ├── db_handler.py       # Database operations
    └── rag_search.py       # Semantic search
```

## How It Works

1. **Video Check**: When you provide a YouTube URL, it checks if the video is already in the database
2. **Processing**: If new, it downloads audio, transcribes, chunks, and creates embeddings
3. **Storage**: All data is stored in SQLite for future queries
4. **RAG Search**: When you ask a question, it performs semantic search on the chunks
5. **Response**: Returns the most relevant sections from the video

## Commands

- Type your message normally to chat
- Provide a YouTube URL to load a video
- Ask questions about loaded videos
- Type `exit`, `quit`, or `bye` to end the session
