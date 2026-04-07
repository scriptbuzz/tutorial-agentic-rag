import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChuncker:
    """
    Implements a hierarchical (Parent-Child) document splitting strategy.
    
    1. Parent Chunks: Split based on Markdown headers (H1, H2, H3) to preserve semantic structure.
    2. Child Chunks: Sub-split parents into small, fixed-size windows for high-precision vector search.
    
    This allows the system to search small snippets but provide the LLM with large, 
    contextually rich parent documents.
    """
    def __init__(self):
        # The parent splitter looks for structural Markdown boundaries
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        # The child splitter breaks parents into small overlapping windows
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        """Processes a single file into a hierarchy of documents."""
        doc_path = Path(md_path)
        
        # 1. Initial split by Markdown headers
        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())
        
        # 2. Refine parents: merge tiny headers and split massive ones
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        
        all_parent_chunks, all_child_chunks = [], []
        # 3. Create the many-to-one relationship between children and parents
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path)
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):
        """
        Groups tiny split segments together if they are smaller than 
        the MIN_PARENT_SIZE. This prevents having too many tiny context
        windows that might lack enough text for the LLM to reason with.
        """
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged

    def __split_large_parents(self, chunks):
        """
        If a parent section is extremely large (e.g., a massive H1 section
        without H2s), it splits it down to MAX_PARENT_SIZE to avoid 
        blowing out the LLM's context window.
        """
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def __clean_small_chunks(self, chunks):
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        """
        The final phase where 'Children' are generated from 'Parents'.
        Each child gets a reference back to the Parent ID so that the 
        Agent can pull the full context during retrieval.
        """
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": str(doc_path.stem)+".pdf", "parent_id": parent_id})
            
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))