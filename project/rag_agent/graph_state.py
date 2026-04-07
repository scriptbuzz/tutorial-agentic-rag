from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """
    State reducer that either appends new agent answers to the list OR
    resets the entire list to empty if the '__reset__' flag is detected.
    Used when starting a brand new research loop for a query.
    """
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

class State(MessagesState):
    """
    The global state object for the parent LangGraph.
    This persists throughout the entire conversation pipeline.
    """
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = "" 
    rewrittenQuestions: List[str] = []
    # Accumulate answers from multiple parallel agents
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """
    The local state object for individual worker agents.
    Each parallel sub-query research agent gets its own copy of this state.
    """
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    # Store search/retrieve unique identifiers to avoid redundant work
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []
    
    # Internal counters for circuit-breaking
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0