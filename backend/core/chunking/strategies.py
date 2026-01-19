"""
Chunking Strategies for RAG Pipeline.

FOCUS: Recursive chunking (512 tokens, 50 overlap)
MUST: Test page-level for highest accuracy
OPTIONS: Fixed, Semantic, Sentence, Document

Default recommendation: Start with recursive chunking at 512 tokens.
Test page-level chunking if documents have clear page boundaries.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence
import numpy as np


import tiktoken

from .preprocessor import TextPreprocessor

class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED= "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    DOCUMENT = "document"
    PAGE  = "page"
    

@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    
    Contains the chunk content and metadata for tracking
    source, position, and relationships.
    """
    
    content: str
    chunk_id: str = ""
    
    # Source tracking
    document_id: str = ""
    source_file: str = ""
    
    # Position tracking
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    
    # Optional metadata
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    # Token count (populated after tokenization)
    token_count: int = 0
    
    def __post__init__(self):
        """Generate chunk ID if not provided."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
            
    def _generate_id(self) -> str:
        """Generate a unique chunk id based on content and position."""
        content_hash = hashlib.md5(
            f"{self.document_id}:{self.start_char}:{self.content[:100]}".encode()
        ).hexdigest()[:12]
        return f"chunk_{content_hash}"
    
    @property
    def char_count(self) -> int:
        """Character count of the chunk."""
        return len(self.content)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number,
            "section": self.section,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
    

class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    
    All chunkers must implement the chunk() method.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        tokenizer: Optional[Callable[[str], list]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
            min_chunk_size: Minimum chunk size in tokens (default: 100)
            tokenizer: Function to tokenize text (returns list of tokens)
            preprocessor: Text preprocessor instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tokenizer or self._default_tokenizer
        self.preprocessor = preprocessor or TextPreprocessor()
        # Cache tiktoken encoding for efficiency
        self._encoding = tiktoken.get_encoding("cl100k_base")
        
    def _default_tokenizer(self, text: str) -> list[int]:
        """
        Tokenizer using tiktoken (cl100k_base encoding).

        Uses the same tokenizer as GPT-4 and GPT-3.5-turbo models.
        """
        return self._encoding.encode(text)

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer(text))
    
    def preprocess(self, text: str) -> str:
        """Preprocess text before chunking."""
        return self.preprocessor.preprocess(text)
    
    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            document_id: Unique document identifier
            source_file: Source file path/name
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _create_chunks_with_metadata(
        self,
        texts: list[str],
        document_id: str,
        source_file: str,
        metadata: Optional[dict],
        start_positions: Optional[list[int]] = None,
    ) -> list[Chunk]:
        """
        Create Chunk objects from text list with proper metadata.
        """
        chunks = []
        total = len(texts)
        
        for i , text in enumerate(texts):
            if not text.strip():
                continue
            
            start_char = start_positions[i] if start_positions else 0
            end_char = start_char + len(texts)
            
            
            chunk = Chunk(
                content= text,
                document_id= document_id,
                source_file= source_file,
                start_char= start_char,
                end_char= end_char,
                chunk_index=i,
                total_chunks= total,
                token_count= self.count_tokens(text),
                metadata= metadata or {},
            )
            chunks.append(chunk)
            
            
        # Update total chunks after filtering empty 
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            
        return chunks
    
    
class FixedChunker(BaseChunker):
    """
    Fixed-size chunking strategy.
    
    Splits text into fixed-size chunks based on token count.
    Simple but can split in the middle of sentences.
    """
    
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:   
        """Split text into fixed-size chunks""" 
        
        text = self.preprocess(text)
        if not text:
            return []
        
        tokens = self.tokenizer(text)
        chunks_text = []
        start_positions = []
        
        i = 0
        char_pos = 0
        
        while i < len(tokens):
            # Get chunk tokens
            chunk_tokens = tokens[i: i + self.chunk_size]
            chunk_text = self._decode_tokens(chunk_tokens)

            if len(chunk_tokens) >= self.min_chunk_size or i + len(chunk_tokens) >= len(tokens):
                chunks_text.append(chunk_text)
                start_positions.append(char_pos)

            # Move forward with overlap
            step = max(1, self.chunk_size - self.chunk_overlap)
            char_pos += len(self._decode_tokens(tokens[i:i + step]))
            i += step
        
        return self._create_chunks_with_metadata(
            chunk_text, document_id, source_file, metadata, start_positions
        )
        
    
class RecursiveChunker(BaseChunker):
    """
    Recursive chunking strategy (RECOMMENDED DEFAULT).
    
    FOCUS: 512 tokens with 50 overlap
    
    Recursively splits text using a hierarchy of separators,
    trying to keep semantically related content together.
    
    Separator hierarchy (default):
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Periods (sentences)
    4. Spaces (words)
    5. Empty string (characters)
    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.separators = separators or self.DEFAULT_SEPARATORS
        
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Recursively split text using separator hierarchy."""
        text = self.preprocess(text)
        if not text:
            return []
        
        chunk_text = self._recursive_split(text, self.separators)
        
        # Merge small chunks and handle overlap
        merged_chunks = self._merge_chunks(chunk_text)
        
        return self._create_chunks_with_metadata(
            merged_chunks, document_id, source_file, metadata
        )
    
    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separators
        if separator:
            splits = text.split(separator)
            # Re-add separator to end of each split (except last)
            splits= [s + separator if i < len(splits) - 1 else s
                     for i, s in enumerate(splits)]
        
        else:
            # Character-level split as last resot 
            return [text[i:i+ self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]
            
            
        chunks= []
        for split in splits:
            if not split.strip():
                continue
            
            token_count = self.count_tokens(split)
            
            if token_count <= self.chunk_size:
                chunks.append(split)
            elif remaining_separators:
                # Recursive split with next separator
                sub_chunks = self._recursive_split(split, remaining_separators) 
                chunks.extend(sub_chunks)
            else:
                # Last resort : force split 
                chunks.append(split[:self.chunk_size * 4]) # Approximate chars
        
        return chunks 
    
    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and add overlap"""
        if not chunks:
            return []
        
        merged = []
        current = ""
        
        for chunk in chunks:
            combined = current + chunk if current else chunk
            combined_tokens = self.count_tokens(combined)
            
            if combined_tokens <= self.chunk_size:
                current = combined
                
            else: 
                if current:
                    merged.append(current.strip())
                current = chunk
        
        if current:
            merged.append(current.strip())
            
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(merged) > 1:
            merged = self._add_overlap(merged)
                
        return merged            

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap from previous chunk to each chunk."""
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_tokens = self.tokenizer(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap:]
            overlap_text = self._decode_tokens(overlap_tokens)

            result.append(overlap_text + ' ' + chunks[i])

        return result
    
class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking strategy.
    
    Splits text into sentences first, then groups sentences
    to meet target chunk size. Ensures chunks end at sentence boundaries.
    """
    
    # Sentence-ending pattern
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$',
        re.MULTILINE
    )
    
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split text into sentence-based chunks."""
        text = self.preprocess(text)
        if not text:
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Group sentences into chunks
        chunks_text = self._group_sentences(sentences)
        
        return self._create_chunks_with_metadata(
            chunks_text, document_id, source_file, metadata
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Use regex to split on sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks of target size."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Handle sentence longer than chunk_size
                if sentence_tokens > self.chunk_size:
                    # Split long sentence (fallback to recursive)
                    recursive = RecursiveChunker(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    sub_chunks = recursive._recursive_split(sentence, [", ", " ", ""])
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_tokens = 0
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy.
    
    Uses embeddings to detect semantic boundaries in text.
    Groups semantically similar content together.
    
    Requires an embedding function to be provided.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
        embedding_func: Optional[Callable[[list[str]], list[list[float]]]] = None,
        **kwargs,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_func = embedding_func
    
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split text at semantic boundaries."""
        text = self.preprocess(text)
        if not text:
            return []
        
        if self.embedding_func is None:
            # Fallback to sentence chunking if no embedding function
            fallback = SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                tokenizer=self.tokenizer,
            )
            return fallback.chunk(text, document_id, source_file, metadata)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return self._create_chunks_with_metadata(
                [text], document_id, source_file, metadata
            )
        
        # Get embeddings for each sentence
        embeddings = self.embedding_func(sentences)
        
        # Find semantic boundaries
        boundaries = self._find_boundaries(embeddings)
        
        # Create chunks based on boundaries
        chunks_text = self._create_semantic_chunks(sentences, boundaries)
        
        return self._create_chunks_with_metadata(
            chunks_text, document_id, source_file, metadata
        )
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_boundaries(self, embeddings: list[list[float]]) -> list[int]:
        """
        Find semantic boundaries by detecting drops in similarity.
        
        Returns indices where new chunks should start.
        """
        
        boundaries = [0]
        
        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(
                embeddings[i - 1], embeddings[i]
            )
            
            if similarity < self.similarity_threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_semantic_chunks(
        self,
        sentences: list[str],
        boundaries: list[int],
    ) -> list[str]:
        """Create chunks based on semantic boundaries."""
        chunks = []
        
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            
            # Check if chunk exceeds size limit
            if self.count_tokens(chunk_text) > self.chunk_size:
                # Split further using recursive chunking
                recursive = RecursiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    tokenizer=self.tokenizer,
                )
                sub_chunks = recursive._recursive_split(
                    chunk_text, [". ", ", ", " "]
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks


class DocumentChunker(BaseChunker):
    """
    Document-level chunking strategy.
    
    Treats the entire document as a single chunk.
    Useful for small documents or when full context is needed.
    """
    
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Return entire document as single chunk."""
        text = self.preprocess(text)
        if not text:
            return []
        
        return self._create_chunks_with_metadata(
            [text], document_id, source_file, metadata
        )
    
class PageChunker(BaseChunker):
    """
    Page-level chunking strategy.
    
    MUST: Test for highest accuracy with PDF documents.
    
    Splits text by page markers. Each page becomes a chunk,
    with optional overlap between pages.
    """
    
    # Common page markers
    PAGE_MARKERS = [
        r'\f',  # Form feed
        r'---\s*Page\s*\d+\s*---',
        r'\[Page\s*\d+\]',
        r'Page\s*\d+\s*of\s*\d+',
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        page_separator: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.page_separator = page_separator
        
    
    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split text by page boundaries."""
        text = self.preprocess(text)
        if not text:
            return []
        
        # Split by page markers
        pages = self._split_by_pages(text)
        
        chunks = []
        for i , page_text in enumerate(pages):
            if not page_text.strip():
                continue

            # Check if page exceeds chunk size
            if self.count_tokens(page_text) > self.chunk_size:
                # Split page using recursive chunking
                recursive = RecursiveChunker(
                    chunk_size= self.chunk_size,
                    chunk_overlap= self.chunk_overlap,
                    tokenizer = self.tokenizer
                )
                page_chunks = recursive.chunk(
                    page_text, document_id, source_file,
                    {**(metadata or {}), "page_number": i + 1}
                )
                for pc in page_chunks:
                    pc.page_number = i + 1
                chunks.extend(page_chunks)
            else:
                chunk = Chunk(
                    content=page_text,
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=len(chunks),
                    page_number=i + 1,
                    token_count=self.count_tokens(page_text),
                    metadata=metadata or {},
                )
                chunks.append(chunk)
                
        # Updae total chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)
            
        return chunks
    
    def _split_by_pages(self, text: str) -> list[str]:
        """Split text by page markers."""
        if self.page_separator:
            return text.split(self.page_separator)
        
        # Try common page markers
        for marker in self.PAGE_MARKERS:
            pattern = re.compile(marker, re.IGNORECASE)
            if pattern.search(text):
                return pattern.split(text)
            
        # Fallback: return whole text
        
        return [text]
    
    
def get_chunker(
    strategy: ChunkingStrategy | str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> BaseChunker:
    """
    Factory function to get a chunker by strategy name.
    
    Args:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in tokens (default: 512)
        chunk_overlap: Overlap between chunks (default: 50)
        **kwargs: Additional arguments for specific chunkers
        
    Returns:
        Configured chunker instance
        
    Example:
        >>> chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk(text, document_id="doc123")
    """
    if isinstance(strategy, str):
        strategy = ChunkingStrategy(strategy.lower())
    
    chunkers = {
        ChunkingStrategy.FIXED: FixedChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.DOCUMENT: DocumentChunker,
        ChunkingStrategy.PAGE: PageChunker,
    }
    
    chunker_class = chunkers.get(strategy)
    if not chunker_class:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return chunker_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )

