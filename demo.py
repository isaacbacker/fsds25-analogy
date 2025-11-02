"""
Quick demonstration of the analogy package capabilities.

This script shows the main features without requiring model downloads or API keys.
"""

from analogy import WordEmbeddings, LLMAnalogySolver


def demo_embeddings():
    """Demonstrate WordEmbeddings features."""
    print("=" * 70)
    print("WordEmbeddings Demonstration")
    print("=" * 70)
    
    # Initialize
    embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-100")
    
    print("\n📊 Available Pre-trained Models:")
    print("-" * 70)
    models = embeddings.list_available_models()
    print(f"Total models available: {len(models)}")
    print("\nSample models:")
    for i, model in enumerate(models[:5], 1):
        print(f"  {i}. {model}")
    
    print("\n💡 How Word Embeddings Work:")
    print("-" * 70)
    print("Word embeddings represent words as dense vectors (arrays of numbers).")
    print("Words with similar meanings have similar vectors.")
    print("")
    print("Example: 'king' might be represented as:")
    print("  [0.50, -0.32, 0.87, 0.12, ..., 0.45]  (100-300 dimensions)")
    print("")
    print("Analogies work through vector arithmetic:")
    print("  vector(king) - vector(man) + vector(woman) ≈ vector(queen)")
    
    print("\n🎯 Example Analogies to Try:")
    print("-" * 70)
    print("Once you download a model with embeddings.download_model():")
    print("")
    print("  # Gender relationships")
    print("  embeddings.solve_analogy('man', 'woman', 'king')")
    print("  # → Expected: queen")
    print("")
    print("  # Geography")
    print("  embeddings.solve_analogy('paris', 'france', 'london')")
    print("  # → Expected: england")
    print("")
    print("  # Verb tenses")
    print("  embeddings.solve_analogy('walk', 'walked', 'swim')")
    print("  # → Expected: swam")
    
    print("\n📝 Note:")
    print("-" * 70)
    print("To actually run analogies, you need to download a model first:")
    print("  embeddings.download_model()")
    print("This will download and cache the model (may take a few minutes).")
    print("")
    print("Run: python examples/basic_embeddings.py")
    print("for a complete working example.")


def demo_llm():
    """Demonstrate LLMAnalogySolver features."""
    print("\n" + "=" * 70)
    print("LLMAnalogySolver Demonstration")
    print("=" * 70)
    
    print("\n🤖 How LLM Analogies Work:")
    print("-" * 70)
    print("Large Language Models (like GPT) solve analogies using:")
    print("  • Pattern recognition from training data")
    print("  • Contextual understanding of relationships")
    print("  • Semantic reasoning about language")
    print("")
    print("Unlike embeddings, LLMs can also explain WHY an analogy works.")
    
    print("\n🎯 Example Usage:")
    print("-" * 70)
    print("With an OpenAI API key:")
    print("")
    print("  llm = LLMAnalogySolver(model='gpt-3.5-turbo')")
    print("  answers = llm.solve_analogy('man', 'woman', 'king')")
    print("  # → ['queen', 'monarch', 'ruler', ...]")
    print("")
    print("  explanation = llm.explain_analogy('man', 'woman', 'king', 'queen')")
    print("  # → 'The relationship between man and woman is one of gender...'")
    
    print("\n🔑 Setup Required:")
    print("-" * 70)
    print("1. Get an API key from https://platform.openai.com/api-keys")
    print("2. Set environment variable:")
    print("     export OPENAI_API_KEY='your-key-here'")
    print("   OR create a .env file:")
    print("     OPENAI_API_KEY=your-key-here")
    print("")
    print("Run: python examples/llm_analogies.py")
    print("for a complete working example.")


def demo_comparison():
    """Compare the two approaches."""
    print("\n" + "=" * 70)
    print("Embeddings vs LLM: Which to Use?")
    print("=" * 70)
    
    print("\n📊 Embeddings (Word2Vec, GloVe, FastText)")
    print("-" * 70)
    print("Pros:")
    print("  ✓ Fast (microseconds per query)")
    print("  ✓ Deterministic (same input → same output)")
    print("  ✓ Works offline after download")
    print("  ✓ Free to use")
    print("  ✓ Provides similarity scores")
    print("")
    print("Cons:")
    print("  ✗ Limited to vocabulary (can't handle unknown words)")
    print("  ✗ No explanations")
    print("  ✗ Less flexible with complex relationships")
    
    print("\n🤖 LLMs (GPT-3.5, GPT-4, etc.)")
    print("-" * 70)
    print("Pros:")
    print("  ✓ Handles complex, nuanced relationships")
    print("  ✓ Can explain reasoning")
    print("  ✓ More flexible and creative")
    print("  ✓ Better context understanding")
    print("")
    print("Cons:")
    print("  ✗ Requires API key and internet")
    print("  ✗ Slower (100s of milliseconds)")
    print("  ✗ Costs money per API call")
    print("  ✗ May be non-deterministic")
    
    print("\n💡 Recommendations:")
    print("-" * 70)
    print("Use EMBEDDINGS for:")
    print("  • Research and exploration")
    print("  • Batch processing many analogies")
    print("  • Offline applications")
    print("  • When you need precise similarity scores")
    print("")
    print("Use LLMs for:")
    print("  • Complex or creative analogies")
    print("  • When you need explanations")
    print("  • Interactive applications")
    print("  • Educational purposes")
    
    print("\n🔬 Try Both!")
    print("-" * 70)
    print("Run: python examples/compare_methods.py")
    print("to see both approaches side-by-side with real examples.")


def main():
    """Run the complete demonstration."""
    print("\n" + "=" * 70)
    print("FSDS25 Word Analogies Package - Quick Demo")
    print("=" * 70)
    print("\nThis package helps you explore word analogies like:")
    print("  'man' : 'woman' :: 'king' : 'queen'")
    print("\nUsing two powerful approaches:")
    print("  1. Word embeddings (Word2Vec, GloVe)")
    print("  2. Large Language Models (GPT)")
    
    demo_embeddings()
    demo_llm()
    demo_comparison()
    
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Try the examples:")
    print("     python examples/basic_embeddings.py")
    print("     python examples/llm_analogies.py")
    print("     python examples/compare_methods.py")
    print("")
    print("2. Read the documentation:")
    print("     cat README.md")
    print("")
    print("3. Run the tests:")
    print("     pytest tests/")
    print("")
    print("4. Experiment with your own analogies!")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
