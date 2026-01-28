"""
Chunking Strategies for RAG Pipeline.

FOCUS: Recursive chunking (512 tokens, 50 overlap)
MUST: Test page-level for highest accuracy
OPTIONS: Fixed, Semantic, Sentence, Document, Page

Uses langchain-text-splitters for robust implementations.
Install: pip install langchain-text-splitters tiktoken
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import tiktoken
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)

from ..config import settings
from .preprocessor import TextPreprocessor


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    DOCUMENT = "document"
    PAGE = "page"


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    chunk_id: str = ""
    document_id: str = ""
    source_file: str = ""
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        content_hash = hashlib.md5(
            f"{self.document_id}:{self.start_char}:{self.content[:100]}".encode()
        ).hexdigest()[:12]
        return f"chunk_{content_hash}"

    def to_dict(self) -> dict:
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


class Chunker:
    """
    Text chunker using LangChain text splitters.

    FOCUS: 512 tokens with 50 overlap (recursive recommended)

    Usage:
        chunker = Chunker(strategy="recursive", chunk_size=512)
        chunks = chunker.chunk(text, document_id="doc123")
    """

    # Token encoding for counting
    _encoding = tiktoken.get_encoding("cl100k_base")

    def __init__(
        self,
        strategy: Optional[ChunkingStrategy | str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        page_separator: Optional[str] = None,
        embedding_func: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ):
        """
        Initialize chunker.

        Args:
            strategy: Chunking strategy to use (defaults to config)
            chunk_size: Target chunk size in tokens (defaults to config)
            chunk_overlap: Overlap between chunks in tokens (defaults to config)
            page_separator: Custom page separator for page-level chunking
            embedding_func: Embedding function for semantic chunking
        """
        # Use config defaults if not specified
        if strategy is None:
            strategy = settings.chunking.strategy
        if chunk_size is None:
            chunk_size = settings.chunking.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunking.chunk_overlap

        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy.lower())

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.page_separator = page_separator
        self.embedding_func = embedding_func
        self.preprocessor = TextPreprocessor()

        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create appropriate LangChain splitter for strategy."""
        # Convert token count to approximate char count (avg 4 chars per token)
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4

        if self.strategy == ChunkingStrategy.FIXED:
            return TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=char_size,
                chunk_overlap=char_overlap,
                length_function=self._count_tokens,
                separators=["\n\n", "\n", ". ", ", ", " ", ""],
            )

        elif self.strategy == ChunkingStrategy.SENTENCE:
            return RecursiveCharacterTextSplitter(
                chunk_size=char_size,
                chunk_overlap=char_overlap,
                length_function=self._count_tokens,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
            )

        elif self.strategy == ChunkingStrategy.SEMANTIC:
            # Use sentence-transformers token splitter for semantic-aware splitting
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap,
                tokens_per_chunk=self.chunk_size,
            )

        elif self.strategy == ChunkingStrategy.PAGE:
            separator = self.page_separator or "\f"
            return CharacterTextSplitter(
                separator=separator,
                chunk_size=char_size,
                chunk_overlap=char_overlap,
                length_function=self._count_tokens,
            )

        elif self.strategy == ChunkingStrategy.DOCUMENT:
            # Document strategy returns the whole text
            return None

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))

    def chunk(
        self,
        text: str,
        document_id: str = "",
        source_file: str = "",
        metadata: Optional[dict] = None,
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
        # Preprocess text
        text = self.preprocessor.preprocess(text)
        if not text:
            return []

        # Document strategy: return whole text as single chunk
        if self.strategy == ChunkingStrategy.DOCUMENT:
            return [
                Chunk(
                    content=text,
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=0,
                    total_chunks=1,
                    token_count=self._count_tokens(text),
                    metadata=metadata or {},
                )
            ]

        # Use LangChain splitter
        texts = self._splitter.split_text(text)

        # Convert to Chunk objects
        chunks = []
        for i, chunk_text in enumerate(texts):
            if not chunk_text.strip():
                continue

            chunk = Chunk(
                content=chunk_text,
                document_id=document_id,
                source_file=source_file,
                chunk_index=i,
                total_chunks=len(texts),
                token_count=self._count_tokens(chunk_text),
                metadata=metadata or {},
            )

            # For page strategy, set page number
            if self.strategy == ChunkingStrategy.PAGE:
                chunk.page_number = i + 1

            chunks.append(chunk)

        # Update total_chunks after filtering empty
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks


def get_chunker(
    strategy: Optional[ChunkingStrategy | str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    **kwargs,
) -> Chunker:
    """
    Factory function to get a chunker by strategy name.

    Args:
        strategy: Chunking strategy to use (defaults to config)
        chunk_size: Target chunk size in tokens (defaults to config)
        chunk_overlap: Overlap between chunks (defaults to config)
        **kwargs: Additional arguments for specific strategies

    Returns:
        Configured Chunker instance

    Example:
        >>> chunker = get_chunker()  # Uses config settings
        >>> chunks = chunker.chunk(text, document_id="doc123")
    """
    return Chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
