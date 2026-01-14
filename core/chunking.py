from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class RecursiveChunker:
    """Implements recursive chunking with token-aware splitting"""
    
    def __init__(self, chunk_size:int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Recurse separators in order of preference
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",
            "? ",
            "; ",
            ", ",
            " ",    # Words
            ""      # Characters
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = self._token_length,
            separators= self.separators,
            keep_separator=True
        )
        
    def _token_length(self, text:str) -> int :
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_document(self, text: str, metadata: Dict= None) -> List[Dict]:
        """
        Chunk document with metadata preservation
        
        Returns:
            List of chunks with metadata
        """
        chunks = self.splitter.split_text(text)
        
        return [
            {
                "content" : chunk,
                "chunk_id" : f"chunk_{i}",
                "chunk_idx" : i,
                "total_chunks" : len(chunks),
                "token_count" : self._token_length(chunk),
                "metadata" : metadata or {}
            }
            for i, chunk in enumerate(chunks)
        ]
        