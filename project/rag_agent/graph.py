from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *

def create_agent_graph(llm, tools_list):
    """
    Compiles the main application state machine and the worker subgraphs.
    This sets up a Map-Reduce architecture: the main graph maps rewritten queries
    to parallel subgraphs, and then reduces the results in `aggregate_answers`.
    """
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    # In-memory checkpointing saves the State object across continuous
    # message calls, acting as the 'memory' subsystem.
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    # === WORKER SUBGRAPH (Retrieval & Research) ===
    # This subgraph runs for each individual rewritten query.
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)

    # Subgraph control flow limits execution and triggers loops if tools are called.
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    # === MAIN GRAPH (Orchestration & Aggregation) ===
    # This acts as the parent workflow defining the macroscopic user interaction.
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    # Define the topological edges for the parent nodes
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    
    # Send agent output to the aggregation layer
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    # Compile with human-in-the-loop interruption enabled. 
    # If the system ends up at `request_clarification`, it halts execution here and waits.
    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    print("✓ Agent graph compiled successfully.")
    return agent_graph