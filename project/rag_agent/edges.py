from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """
    Evaluates the 'questionIsClear' boolean flag set by the rewrite node.
    If the user's question is impossibly vague, it routes to 'request_clarification'.
    Otherwise, it triggers the Map-Reduce pattern by sending multiple 'Send' objects
    to the 'agent' subgraph—spawning parallel workers for each sub-query.
    """
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]
    
def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    """
    Evaluates the output of the Orchestrator LLM.
    If the limiters (iterations/tools) are breached, it forces a fallback mechanism to prevent loops.
    If the LLM provided an unstructured string, it assumes it's the final answer and routes to 'collect_answer'.
    If the LLM invoked a function mapping in its JSON blob, it routes to 'tools'.
    """
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    # Hard circuit breakers to prevent infinite token consumption
    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        # No more search needed, it has the final answer.
        return "collect_answer"
    
    # Needs more data from the DB.
    return "tools"