import uuid
from langchain_ollama import ChatOllama
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability

class RAGSystem:
    """
    The orchestrator class for the entire Agentic RAG engine.
    This acts as the glue tying together external vector databases,
    the LangGraph state machine, and LangChain model bindings.
    """

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        
        # Instantiate dependent system managers
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        
        self.agent_graph = None
        
        # We assign a unique thread_id per session to act as a unique
        # identifier for LangGraph's persistent state checkpointing.
        # This gives the agent its "memory" across messages.
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        """
        Bootstraps all necessary stateful elements:
        1. Ensures the target Qdrant collection actually exists.
        2. Binds the LLM model to be used by the Graph.
        3. Mounts the execution tools onto the Agent graph.
        """
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        # Initialize the target foundational model
        llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        
        # Create function-calling schemas mapped to our retrieval database
        tools = ToolFactory(collection).create_tools()
        
        # Construct and compile the LangGraph workflow
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())