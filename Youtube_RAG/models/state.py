from typing import TypedDict, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]
    youtube_audio: Optional[bytes]
    youtube_transcript: Optional[str]
    youtube_chunks: Optional[list]
    youtube_video_id: Optional[str]
    youtube_url: Optional[str]
    vectors: Optional[list]
    rag_search_results: Optional[list]
