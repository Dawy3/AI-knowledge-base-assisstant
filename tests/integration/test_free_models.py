"""
Test script to verify free models are working with the config system.
"""
import asyncio
from backend.core.config import settings
from backend.core.embedding.generator import create_embedding_generator
from backend.services.external_api.openai_client import OpenAIClient


async def test_embeddings():
    """Test free HuggingFace embeddings."""
    print("\n" + "="*60)
    print("TESTING FREE EMBEDDINGS")
    print("="*60)

    print(f"\nConfig settings:")
    print(f"  Provider: {settings.embedding.model_provider}")
    print(f"  Model: {settings.embedding.model_name}")
    print(f"  Dimensions: {settings.embedding.dimensions}")

    # Create embedding generator using config
    generator = create_embedding_generator()

    print(f"\n  Generator model: {generator.model_id}")
    print(f"  Batch size: {generator.batch_size}")

    # Test embedding generation
    texts = [
        "What is machine learning?",
        "How does RAG work?",
        "Explain embeddings."
    ]

    print(f"\nGenerating embeddings for {len(texts)} texts...")
    try:
        result = await generator.embed_texts(texts)

        print(f"\nResults:")
        print(f"  Texts embedded: {result.texts_count}")
        print(f"  Embedding dimensions: {result.dimensions}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Cost: ${result.estimated_cost:.6f}")
        print(f"  First embedding (first 5 dims): {result.embeddings[0][:5]}")

        print("\n✓ Embeddings test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Embeddings test FAILED: {e}")
        return False


async def test_llm():
    """Test free LLM via OpenRouter."""
    print("\n" + "="*60)
    print("TESTING FREE LLM")
    print("="*60)

    print(f"\nConfig settings:")
    print(f"  Model: {settings.llm.openai_model}")

    # Create LLM client using config (auto-detects OpenRouter from .env)
    client = OpenAIClient()

    print(f"  Client provider: {client.provider}")
    print(f"  Client model: {client.default_model}")
    print(f"  Base URL: {client.base_url}")

    # Test chat generation
    prompt = "What is 2+2? Answer in one word."

    print(f"\nSending prompt: '{prompt}'")
    try:
        response = await client.chat(prompt)

        print(f"\nResponse:")
        print(f"  Model used: {response.model}")
        print(f"  Content: {response.content}")
        print(f"  Tokens: {response.usage.get('total_tokens', 0)}")

        print("\n✓ LLM test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FREE MODELS CONFIGURATION TEST")
    print("="*60)

    # Test embeddings
    embed_ok = await test_embeddings()

    # Test LLM
    llm_ok = await test_llm()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Embeddings: {'✓ PASS' if embed_ok else '✗ FAIL'}")
    print(f"LLM: {'✓ PASS' if llm_ok else '✗ FAIL'}")

    if embed_ok and llm_ok:
        print("\n✓ ALL TESTS PASSED - Free models configured correctly!")
    else:
        print("\n✗ SOME TESTS FAILED - Check errors above")


if __name__ == "__main__":
    asyncio.run(main())
