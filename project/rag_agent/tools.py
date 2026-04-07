from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

class ToolFactory:
    """
    Factory class to instantiate and expose retrieval tools for the LangGraph agents.
    It encapsulates the logic for searching the vector database (Child Chunks)
    and fetching the full associated context (Parent Chunks).
    """
    
    def __init__(self, collection):
        self.collection = collection
        # Handles the retrieval of full parent documents from disk
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """
        Executes a similarity search against the vector database.
        It returns the top-K individual 'Child' text segments.
        If no results are found above the threshold, it flags this for the agent.
        """
        try:
            results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """
        Takes a list of parent IDs and fetches their full text content from disk.
        This provides the final 'Generation' step with the complete context needed 
        to synthesize an accurate answer.
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Fetch the complete context of a parent chunk using its ID.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Instantiate and array tools for agent binding."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]