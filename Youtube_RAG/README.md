# YouTube RAG Assistant

Chat with YouTube videos. Ask questions, get answers. Simple as that.

## What it does

Ever wanted to ask a YouTube video a question instead of scrubbing through timestamps? That's what this does. Drop a URL, ask anything, get answers pulled straight from the transcript.

**How it works:**
- Downloads and transcribes any YouTube video
- Stores everything locally (SQLite)
- Uses semantic search to find relevant parts
- Streams answers in real-time

**Built with:** LangGraph + Google Gemini + RAG

## Quick Start

1. Install stuff:
```bash
pip install -r requirements.txt
```

2. Add your Google API key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

3. Run it:
```bash
python main.py
```

## How to use

Just talk to it naturally:

```
You: https://www.youtube.com/watch?v=arj7oStGLkU

[State Update] Loading existing video data...
   Loaded 109 chunks and 109 vectors

[Assistant]: Video is loaded and ready. What would you like to know?

You: What's inside a procrastinator's mind?

[RAG Search] Searching video content...
   Found 3 relevant sections

[Assistant]: Inside the mind of a procrastinator, there's an instant-gratification
monkey that seeks immediate pleasure. This monkey takes over and prevents them from
reaching important tasks. The video suggests that procrastinators' brains might
actually be different from other people, and mindfulness training can be twice as
effective as standard therapy for breaking these habits...
```

## Commands

- Paste a YouTube URL to load a video
- Ask questions about the video
- Type `exit` or `quit` to leave

That's it. No complicated stuff.

## Project Structure

```
Youtube_RAG/
├── main.py                    # Main chat interface
├── models/state.py            # State management
├── nodes/                     # Graph nodes (agent, processors, RAG)
├── routers/                   # Routing logic
├── tools/                     # LangChain tools
├── services/                  # Video processing pipeline
└── utils/                     # Helper functions
```

## Why this exists

Because sometimes you want to know what's in a 2-hour podcast without watching the whole thing. Or you want to quote something but can't remember where it was said. Or you're just lazy and want a video to answer your questions directly.

Built this to learn LangGraph and mess around with RAG. Turned out pretty useful.
