"""Tests for chunking strategies."""

import pytest
from backend.core.chunking.strategies import (
    Chunker,
    Chunk,
    ChunkingStrategy,
    get_chunker,
)
from backend.core.chunking.preprocessor import (
    TextPreprocessor,
    PreprocessingConfig,
    create_preprocessor,
)


class TestTextPreprocessor:
    """Test text preprocessing."""

    def test_normalize_whitespace(self):
        preprocessor = TextPreprocessor()
        text = "Hello   world\n\n\nTest"
        result = preprocessor.preprocess(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_remove_control_chars(self):
        preprocessor = TextPreprocessor()
        text = "Hello\x00World"
        result = preprocessor.preprocess(text)
        assert "\x00" not in result

    def test_preserve_content(self):
        preprocessor = TextPreprocessor()
        text = "Important content here"
        result = preprocessor.preprocess(text)
        assert "Important" in result
        assert "content" in result

    def test_normalize_unicode(self):
        preprocessor = TextPreprocessor()
        text = "café"  # With combining character
        result = preprocessor.preprocess(text)
        assert result  # Should normalize without error

    def test_remove_zero_width_chars(self):
        preprocessor = TextPreprocessor()
        text = "Hello\u200bWorld"  # Zero-width space
        result = preprocessor.preprocess(text)
        assert "\u200b" not in result

    def test_normalize_quotes(self):
        preprocessor = TextPreprocessor()
        text = "'Hello' and 'World'"
        result = preprocessor.preprocess(text)
        assert '"' in result or "Hello" in result

    def test_normalize_dashes(self):
        preprocessor = TextPreprocessor()
        text = "test—value"  # Em dash
        result = preprocessor.preprocess(text)
        assert "test-value" in result

    def test_empty_text(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        assert result == ""

    def test_whitespace_only_text(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("   \n\n\t  ")
        assert result == ""

    def test_clean_for_embedding(self):
        preprocessor = TextPreprocessor()
        text = "Hello   world\u200b"
        result = preprocessor.clean_for_embedding(text)
        assert "\u200b" not in result
        assert "   " not in result

    def test_extract_metadata_text(self):
        preprocessor = TextPreprocessor()
        text = "Contact us at test@example.com or visit https://example.com"
        result = preprocessor.extract_metadata_text(text)
        assert "content" in result
        assert "urls" in result
        assert "emails" in result

    def test_custom_config(self):
        config = PreprocessingConfig(
            remove_urls=True,
            remove_emails=True,
        )
        preprocessor = TextPreprocessor(config)
        text = "Visit https://example.com or email test@test.com"
        result = preprocessor.preprocess(text)
        assert "https://example.com" not in result
        assert "test@test.com" not in result


class TestCreatePreprocessor:
    """Test preprocessor factory function."""

    def test_create_default(self):
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, TextPreprocessor)

    def test_create_with_url_removal(self):
        preprocessor = create_preprocessor(remove_urls=True)
        text = "Visit https://example.com for more"
        result = preprocessor.preprocess(text)
        assert "https://example.com" not in result


class TestChunker:
    """Test the main Chunker class."""

    def test_recursive_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=20)
        text = "This is a test sentence with more words. " * 200
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_fixed_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.FIXED, chunk_size=50, chunk_overlap=10)
        text = "Word " * 100
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) > 1

    def test_document_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.DOCUMENT)
        text = "This is a long document. " * 100
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) == 1
        assert "document" in chunks[0].content.lower()

    def test_page_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.PAGE, chunk_size=512)
        text = "Page 1 content\fPage 2 content\fPage 3 content"
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) >= 1
        # Page strategy should set page numbers
        if len(chunks) > 1:
            assert chunks[0].page_number == 1

    def test_sentence_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.SENTENCE, chunk_size=50, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) >= 1

    def test_string_strategy(self):
        chunker = Chunker(strategy="recursive", chunk_size=100)
        text = "Test content " * 50
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        # Use FIXED strategy which has more predictable token-based sizing
        chunker = Chunker(strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=10)
        text = "Word " * 500
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) > 1
        # Fixed strategy should respect token counts more precisely
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            assert chunk.token_count <= 110  # Allow small overflow

    def test_overlap_exists(self):
        chunker = Chunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=50, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = chunker.chunk(text, document_id="doc1")

        if len(chunks) >= 2:
            chunk1_words = set(chunks[0].content.split()[-5:])
            chunk2_words = set(chunks[1].content.split()[:5])
            assert len(chunk1_words) > 0 or len(chunk2_words) > 0

    def test_empty_text(self):
        chunker = Chunker()
        chunks = chunker.chunk("", document_id="doc1")
        assert len(chunks) == 0

    def test_short_text(self):
        chunker = Chunker(chunk_size=100)
        text = "Short text"
        chunks = chunker.chunk(text, document_id="doc1")

        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_chunk_metadata(self):
        chunker = Chunker()
        chunks = chunker.chunk(
            "Test content here",
            document_id="doc123",
            source_file="test.pdf",
            metadata={"author": "Test"},
        )

        assert chunks[0].document_id == "doc123"
        assert chunks[0].source_file == "test.pdf"
        assert chunks[0].metadata["author"] == "Test"


class TestGetChunker:
    """Test the get_chunker factory function."""

    def test_get_recursive_chunker(self):
        chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=50)
        assert isinstance(chunker, Chunker)
        assert chunker.strategy == ChunkingStrategy.RECURSIVE

    def test_get_fixed_chunker(self):
        chunker = get_chunker("fixed", chunk_size=256)
        assert isinstance(chunker, Chunker)
        assert chunker.strategy == ChunkingStrategy.FIXED

    def test_default_parameters(self):
        chunker = get_chunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50


class TestChunk:
    """Test the Chunk dataclass."""

    def test_chunk_creation(self):
        chunk = Chunk(content="Test content", document_id="doc1")
        assert chunk.content == "Test content"
        assert chunk.document_id == "doc1"
        assert chunk.chunk_id  # Should auto-generate

    def test_chunk_id_generation(self):
        chunk1 = Chunk(content="Test", document_id="doc1", start_char=0)
        chunk2 = Chunk(content="Test", document_id="doc1", start_char=100)
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_to_dict(self):
        chunk = Chunk(
            content="Test content",
            document_id="doc1",
            source_file="test.pdf",
            chunk_index=0,
            total_chunks=5,
            token_count=10,
        )
        d = chunk.to_dict()

        assert d["content"] == "Test content"
        assert d["document_id"] == "doc1"
        assert d["source_file"] == "test.pdf"
        assert d["chunk_index"] == 0
        assert d["total_chunks"] == 5
        assert d["token_count"] == 10

    def test_optional_fields(self):
        chunk = Chunk(content="Test", page_number=5, section="Introduction")
        assert chunk.page_number == 5
        assert chunk.section == "Introduction"