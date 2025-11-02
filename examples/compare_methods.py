"""
Comparison example showing embeddings vs LLM approaches side-by-side.

This script demonstrates the differences and similarities between
embedding-based and LLM-based analogy solving.
"""

import os
from dotenv import load_dotenv
from analogy import WordEmbeddings, LLMAnalogySolver

load_dotenv()


def compare_methods():
    """Compare embedding-based and LLM-based analogy solving."""
    
    print("=" * 70)
    print("Comparing Embeddings vs LLM for Word Analogies")
    print("=" * 70)
    
    # Define test analogies
    test_cases = [
        ("man", "woman", "king", "queen"),
        ("paris", "france", "london", "england"),
        ("walk", "walked", "swim", "swam"),
    ]
    
    print("\nInitializing models...")
    print("This will download the embedding model (may take a few minutes)...")
    
    # Initialize embeddings
    embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-100")
    embeddings.download_model()
    
    # Initialize LLM (if API key available)
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = LLMAnalogySolver(model="gpt-3.5-turbo")
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Could not initialize LLM: {e}")
    else:
        print("⚠️  OPENAI_API_KEY not found - LLM comparison will be skipped")
    
    # Test each analogy
    for word_a, word_b, word_c, expected in test_cases:
        print("\n" + "=" * 70)
        print(f"Analogy: '{word_a}' : '{word_b}' :: '{word_c}' : ?")
        print(f"Expected answer: {expected}")
        print("=" * 70)
        
        # Embeddings approach
        print("\n📊 EMBEDDINGS APPROACH:")
        print("-" * 70)
        try:
            emb_results = embeddings.solve_analogy(word_a, word_b, word_c, topn=5)
            print("Top 5 predictions:")
            for i, (word, score) in enumerate(emb_results, 1):
                marker = "✓" if word.lower() == expected.lower() else " "
                print(f"{marker} {i}. {word:15s} (similarity: {score:.4f})")
            
            # Check if expected answer is in results
            found = any(word.lower() == expected.lower() for word, _ in emb_results)
            if found:
                print(f"\n✓ Found expected answer '{expected}' in top 5")
            else:
                print(f"\n✗ Expected answer '{expected}' not in top 5")
                
        except ValueError as e:
            print(f"❌ Error: {e}")
        
        # LLM approach
        if llm:
            print("\n🤖 LLM APPROACH:")
            print("-" * 70)
            try:
                llm_results = llm.solve_analogy(word_a, word_b, word_c, num_answers=5)
                print("Top 5 predictions:")
                for i, word in enumerate(llm_results, 1):
                    marker = "✓" if word.lower() == expected.lower() else " "
                    print(f"{marker} {i}. {word}")
                
                # Check if expected answer is in results
                found = any(word.lower() == expected.lower() for word in llm_results)
                if found:
                    print(f"\n✓ Found expected answer '{expected}' in top 5")
                else:
                    print(f"\n✗ Expected answer '{expected}' not in top 5")
                
                # Get explanation
                print("\n💡 Explanation:")
                explanation = llm.explain_analogy(word_a, word_b, word_c, expected)
                print(explanation)
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Embeddings vs LLM")
    print("=" * 70)
    
    print("\n📊 EMBEDDINGS (Word2Vec, GloVe, etc.):")
    print("  ✓ Fast and deterministic")
    print("  ✓ Works offline after initial download")
    print("  ✓ Good for standard analogies")
    print("  ✓ Provides similarity scores")
    print("  ✗ Limited by training vocabulary")
    print("  ✗ Cannot explain reasoning")
    
    print("\n🤖 LLM (GPT-3.5, GPT-4, etc.):")
    print("  ✓ Can handle complex relationships")
    print("  ✓ Provides explanations")
    print("  ✓ Understands context better")
    print("  ✓ More flexible and creative")
    print("  ✗ Requires API key and internet")
    print("  ✗ Slower and has usage costs")
    print("  ✗ May be non-deterministic")
    
    print("\n💡 WHEN TO USE EACH:")
    print("  - Use EMBEDDINGS for: Fast batch processing, offline use, research")
    print("  - Use LLM for: Complex analogies, explanations, interactive applications")
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    compare_methods()
