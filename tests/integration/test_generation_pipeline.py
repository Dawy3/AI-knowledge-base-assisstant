"""
Integration tests for end-to-end generation pipeline.

Tests the full flow: Query Classification -> Context Building -> Prompt -> LLM -> Response
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.query.classifier import (
    QueryClassifier,
    QueryCategory,
    ClassificationResult,
)
from backend.core.generation.context_builder import ContextBuilder, ContextConfig
from backend.core.generation.prompt_manager import PromptManager, PromptConfig
from backend.core.generation.llm_client import LLMClient, LLMResponse, ModelTier


# Test data
SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "content": "Python is a high-level programming language known for readability.", "score": 0.95, "document_id": "d1"},
    {"chunk_id": "c2", "content": "Python supports multiple programming paradigms including procedural and OOP.", "score": 0.90, "document_id": "d1"},
    {"chunk_id": "c3", "content": "Machine learning is a subset of AI that learns from data.", "score": 0.85, "document_id": "d2"},
    {"chunk_id": "c4", "content": "Deep learning uses neural networks with multiple layers.", "score": 0.80, "document_id": "d2"},
]


class TestQueryClassification:
    """Test query classification component."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_simple_query_classification(self, classifier):
        """Test simple query is classified correctly."""
        result = classifier.classify("What is Python?")

        assert result.category in [QueryCategory.SIMPLE, QueryCategory.FAQ]
        assert result.confidence > 0.5
        assert result.latency_ms < 50  # Should be fast

    def test_complex_query_classification(self, classifier):
        """Test complex query is classified correctly."""
        query = """
        Compare and contrast the performance characteristics of different
        machine learning algorithms when applied to large-scale text classification
        tasks, considering both accuracy and computational efficiency.
        """
        result = classifier.classify(query)

        assert result.complexity_score > 0.5

    def test_faq_pattern_detection(self, classifier):
        """Test FAQ patterns are detected."""
        result = classifier.classify("How do I get started?")

        assert result.category == QueryCategory.FAQ
        assert result.use_cache is True

    def test_classification_returns_routing_hints(self, classifier):
        """Test classification provides routing hints."""
        result = classifier.classify("What is the meaning of life?")

        assert result.suggested_model is not None
        assert isinstance(result.use_cache, bool)

    def test_ambiguous_query_detection(self, classifier):
        """Test very short queries are flagged as ambiguous."""
        result = classifier.classify("it")

        assert result.category == QueryCategory.AMBIGUOUS
        assert result.needs_clarification is True


class TestContextBuilder:
    """Test context building component."""

    @pytest.fixture
    def context_builder(self):
        config = ContextConfig(
            simple_max_tokens=800,
            complex_max_tokens=4000,
            default_max_tokens=2000,
        )
        return ContextBuilder(config)

    def test_build_context_simple_query(self, context_builder):
        """Test context building for simple queries."""
        context = context_builder.build(SAMPLE_CHUNKS, query_type="simple")

        assert len(context) > 0
        # Simple queries should have limited context
        total_chars = sum(len(c) for c in context)
        assert total_chars <= 800 * 4  # chars_per_token = 4

    def test_build_context_complex_query(self, context_builder):
        """Test context building for complex queries."""
        context = context_builder.build(SAMPLE_CHUNKS, query_type="complex")

        assert len(context) > 0
        # Complex queries get more context
        total_chars = sum(len(c) for c in context)
        assert total_chars <= 4000 * 4

    def test_build_with_sources(self, context_builder):
        """Test context building with source tracking."""
        context, sources = context_builder.build_with_sources(SAMPLE_CHUNKS)

        assert len(context) == len(sources)
        for source in sources:
            assert "document_id" in source
            assert "chunk_id" in source
            assert "score" in source

    def test_format_context_with_sources(self, context_builder):
        """Test formatted context includes source references."""
        formatted = context_builder.format_context(
            SAMPLE_CHUNKS,
            include_sources=True,
        )

        assert "[1]" in formatted
        assert "[2]" in formatted

    def test_empty_chunks_handled(self, context_builder):
        """Test empty chunk list is handled."""
        context = context_builder.build([])

        assert context == []


class TestPromptManager:
    """Test prompt management component."""

    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()

    def test_build_rag_prompt_basic(self, prompt_manager):
        """Test basic RAG prompt building."""
        contexts = ["Python is a programming language.", "It supports OOP."]

        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="What is Python?",
            contexts=contexts,
        )

        assert system_prompt
        assert "Python" in user_prompt
        assert "Context:" in user_prompt
        assert "Question:" in user_prompt

    def test_build_rag_prompt_with_history(self, prompt_manager):
        """Test RAG prompt with conversation history."""
        contexts = ["Python is a programming language."]
        history = [
            {"role": "user", "content": "Tell me about programming"},
            {"role": "assistant", "content": "Programming is..."},
        ]

        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="What is Python?",
            contexts=contexts,
            history=history,
        )

        assert "Previous conversation:" in user_prompt

    def test_build_prompt_no_context(self, prompt_manager):
        """Test prompt when no context is available."""
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="Unknown topic",
            contexts=[],
        )

        assert "No relevant information" in user_prompt

    def test_build_prompt_out_of_scope(self, prompt_manager):
        """Test prompt for out-of-scope queries."""
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="What's the weather?",
            contexts=["Some irrelevant context"],
            query_type="out_of_scope",
        )

        assert "out of scope" in user_prompt.lower()

    def test_prompt_compression(self, prompt_manager):
        """Test that prompts are reasonably compressed."""
        contexts = ["Context " * 100]  # Long context

        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="Question",
            contexts=contexts,
        )

        # Should be truncated
        assert len(user_prompt) < len(contexts[0]) + 500


class TestLLMClient:
    """Test LLM client component."""

    @pytest.fixture
    def llm_client(self):
        return LLMClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_generate_with_mock(self, llm_client):
        """Test generation with mocked API."""
        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {
                "content": "Python is a programming language.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

            response = await llm_client.generate(
                prompt="What is Python?",
                system_prompt="You are helpful.",
            )

            assert response.content == "Python is a programming language."
            assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_tier_based_routing(self, llm_client):
        """Test tier-based model routing."""
        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {"content": "Response", "usage": {}}

            # Tier 1 should use simple model
            await llm_client.generate("Simple query", tier=ModelTier.TIER_1)
            call_args = mock_call.call_args
            assert call_args[1]["model"] == llm_client.tier1_model

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, llm_client):
        """Test fallback model is used on primary failure."""
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Primary failed")
            return {"content": "Fallback response", "usage": {}}

        with patch.object(llm_client, "_call_with_retry", side_effect=mock_call):
            response = await llm_client.generate("Test query")

            assert response.is_fallback is True
            assert response.model == llm_client.fallback_model

    @pytest.mark.asyncio
    async def test_batch_generation(self, llm_client):
        """Test batch generation."""
        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {"content": "Response", "usage": {}}

            prompts = ["Query 1", "Query 2", "Query 3"]
            responses = await llm_client.generate_batch(prompts, max_concurrent=2)

            assert len(responses) == 3


class TestFullGenerationPipeline:
    """End-to-end generation pipeline tests."""

    @pytest.fixture
    def full_pipeline(self):
        """Setup complete generation pipeline."""
        classifier = QueryClassifier()
        context_builder = ContextBuilder()
        prompt_manager = PromptManager()
        llm_client = LLMClient(api_key="test-key")

        return {
            "classifier": classifier,
            "context_builder": context_builder,
            "prompt_manager": prompt_manager,
            "llm_client": llm_client,
        }

    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, full_pipeline):
        """Test full generation pipeline."""
        classifier = full_pipeline["classifier"]
        context_builder = full_pipeline["context_builder"]
        prompt_manager = full_pipeline["prompt_manager"]
        llm_client = full_pipeline["llm_client"]

        # 1. Classify query
        query = "What is Python used for?"
        classification = classifier.classify(query)

        # 2. Build context based on classification
        query_type = "simple" if classification.category == QueryCategory.SIMPLE else "complex"
        context = context_builder.build(SAMPLE_CHUNKS, query_type=query_type)

        # 3. Build prompt
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query=query,
            contexts=context,
        )

        # 4. Generate (mocked)
        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {
                "content": "Python is used for web development, data science, and automation.",
                "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
            }

            # Route to appropriate tier
            tier = ModelTier.TIER_1 if classification.category == QueryCategory.SIMPLE else ModelTier.TIER_3

            response = await llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                tier=tier,
            )

            assert response.content
            assert response.usage

    @pytest.mark.asyncio
    async def test_pipeline_handles_out_of_scope(self, full_pipeline):
        """Test pipeline handles out-of-scope queries gracefully."""
        classifier = full_pipeline["classifier"]
        prompt_manager = full_pipeline["prompt_manager"]
        llm_client = full_pipeline["llm_client"]

        # Configure classifier to detect out-of-scope
        query = "What's the weather today?"
        classification = classifier.classify(query)

        # Build out-of-scope prompt
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query=query,
            contexts=[],
            query_type="out_of_scope" if not classification.is_in_domain else "normal",
        )

        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {
                "content": "I cannot help with that.",
                "usage": {},
            }

            response = await llm_client.generate(user_prompt, system_prompt)
            assert response.content

    @pytest.mark.asyncio
    async def test_pipeline_with_conversation_history(self, full_pipeline):
        """Test pipeline maintains conversation context."""
        prompt_manager = full_pipeline["prompt_manager"]
        llm_client = full_pipeline["llm_client"]

        history = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query="What are its main features?",
            contexts=["Python supports multiple paradigms."],
            history=history,
        )

        assert "Previous conversation:" in user_prompt

        with patch.object(llm_client, "_call_with_retry") as mock_call:
            mock_call.return_value = {"content": "Its main features include...", "usage": {}}
            response = await llm_client.generate(user_prompt, system_prompt)
            assert response.content


class TestGenerationQuality:
    """Test generation quality aspects."""

    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()

    def test_prompt_includes_context(self, prompt_manager):
        """Test that context is properly included in prompt."""
        contexts = ["Important fact 1.", "Important fact 2."]

        _, user_prompt = prompt_manager.build_rag_prompt(
            query="Question",
            contexts=contexts,
        )

        assert "Important fact 1" in user_prompt
        assert "Important fact 2" in user_prompt

    def test_prompt_is_concise(self, prompt_manager):
        """Test prompts don't have unnecessary bloat."""
        system_prompt, _ = prompt_manager.build_rag_prompt(
            query="Simple question",
            contexts=["Simple context."],
        )

        # System prompt should be concise
        assert len(system_prompt) < 200

    def test_context_builder_respects_limits(self):
        """Test context builder respects token limits."""
        config = ContextConfig(simple_max_tokens=100)
        builder = ContextBuilder(config)

        # Create many long chunks
        long_chunks = [
            {"content": "Word " * 200, "chunk_id": f"c{i}"}
            for i in range(10)
        ]

        context = builder.build(long_chunks, query_type="simple")

        total_chars = sum(len(c) for c in context)
        # Should respect limit (100 tokens * 4 chars/token = 400 chars)
        assert total_chars <= 100 * 4 + 100  # Some buffer for truncation


class TestErrorHandling:
    """Test error handling in generation pipeline."""

    @pytest.mark.asyncio
    async def test_llm_timeout_handling(self):
        """Test LLM client handles timeouts."""
        client = LLMClient(api_key="test", timeout=0.001)

        with patch.object(client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.side_effect = asyncio.TimeoutError()
            mock_get.return_value = mock_client

            with pytest.raises(Exception):
                await client.generate("Test query")

    def test_classifier_handles_empty_query(self):
        """Test classifier handles empty queries."""
        classifier = QueryClassifier()

        result = classifier.classify("")

        assert result.category == QueryCategory.AMBIGUOUS

    def test_context_builder_handles_malformed_chunks(self):
        """Test context builder handles malformed input."""
        builder = ContextBuilder()

        # Missing content key
        chunks = [{"chunk_id": "c1"}]
        context = builder.build(chunks)

        # Should not crash
        assert context == [""]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
