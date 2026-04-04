# agents/state.py
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The unified blackboard for the Multi-Agent Financial Intelligence System.
    """
    
    # --- The Core Chat Log ---
    # `add_messages` is a built-in LangGraph reducer. 
    # When an agent returns `{"messages": [new_message]}`, LangGraph automatically 
    # appends it to this list rather than overwriting the whole history.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # --- Routing & Control Flags ---
    # Tells the graph.py router who should speak next
    next_speaker: str 
    
    # Optional: You can keep high-level status flags if your orchestrator needs them
    # to know when to force-stop the graph, but they are no longer used for 
    # passing the actual financial data.
    status: str 
    is_valid: bool