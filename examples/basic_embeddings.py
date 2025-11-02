"""
Basic example demonstrating word analogies using embeddings.

This script shows how to:
1. Download a pre-trained word embedding model
2. Solve classic analogies like "man:woman::king:queen"
3. Explore word similarities and relationships
"""

from analogy import WordEmbeddings


def main():
    # Initialize the embeddings model
    # Note: For faster testing, you can use smaller models like:
    # - 'glove-wiki-gigaword-100' (smaller, faster)
    # - 'glove-wiki-gigaword-50' (even smaller)
    print("=" * 60)
    print("Word Analogies with Embeddings Example")
    print("=" * 60)
    
    embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-100")
    
    # Download and load the model (this will be cached for future use)
    embeddings.download_model()
    
    print("\n" + "=" * 60)
    print("Example 1: Classic Analogy - Gender")
    print("=" * 60)
    
    # Classic analogy: man is to woman as king is to ?
    print("\nSolving: 'man' is to 'woman' as 'king' is to ?")
    results = embeddings.solve_analogy("man", "woman", "king", topn=5)
    
    print("\nTop 5 answers:")
    for i, (word, score) in enumerate(results, 1):
        print(f"{i}. {word} (similarity: {score:.4f})")
    
    # Explain the analogy
    print("\nAnalyzing the analogy: man:woman::king:queen")
    explanation = embeddings.analogy_explanation("man", "woman", "king", "queen")
    print(f"Relationship similarity: {explanation['relationship_similarity']:.4f}")
    print(f"Prediction quality: {explanation['predicted_similarity']:.4f}")
    print(f"Explanation: {explanation['explanation']}")
    
    print("\n" + "=" * 60)
    print("Example 2: Geographic Analogy")
    print("=" * 60)
    
    # Geographic analogy: paris is to france as london is to ?
    print("\nSolving: 'paris' is to 'france' as 'london' is to ?")
    results = embeddings.solve_analogy("paris", "france", "london", topn=5)
    
    print("\nTop 5 answers:")
    for i, (word, score) in enumerate(results, 1):
        print(f"{i}. {word} (similarity: {score:.4f})")
    
    print("\n" + "=" * 60)
    print("Example 3: Verb Tense Analogy")
    print("=" * 60)
    
    # Verb tense analogy: walking is to walked as swimming is to ?
    print("\nSolving: 'walking' is to 'walked' as 'swimming' is to ?")
    results = embeddings.solve_analogy("walking", "walked", "swimming", topn=5)
    
    print("\nTop 5 answers:")
    for i, (word, score) in enumerate(results, 1):
        print(f"{i}. {word} (similarity: {score:.4f})")
    
    print("\n" + "=" * 60)
    print("Example 4: Word Similarities")
    print("=" * 60)
    
    # Find similar words
    word = "computer"
    print(f"\nWords most similar to '{word}':")
    similar_words = embeddings.most_similar(word, topn=5)
    
    for i, (word, score) in enumerate(similar_words, 1):
        print(f"{i}. {word} (similarity: {score:.4f})")
    
    print("\n" + "=" * 60)
    print("Example 5: Word Pair Similarities")
    print("=" * 60)
    
    # Calculate similarities between word pairs
    word_pairs = [
        ("dog", "cat"),
        ("king", "queen"),
        ("happy", "sad"),
        ("computer", "laptop")
    ]
    
    print("\nSimilarity scores between word pairs:")
    for word1, word2 in word_pairs:
        similarity = embeddings.similarity(word1, word2)
        print(f"  {word1} <-> {word2}: {similarity:.4f}")
    
    print("\n" + "=" * 60)
    print("Understanding Embeddings")
    print("=" * 60)
    
    print("\nWhat is a word embedding?")
    print("A word embedding is a dense vector representation of a word.")
    print("Words with similar meanings have similar vector representations.")
    
    # Show embedding dimensions
    word = "king"
    embedding = embeddings.get_embedding(word)
    if embedding is not None:
        print(f"\nThe word '{word}' is represented as a vector with {len(embedding)} dimensions")
        print(f"First 10 dimensions: {embedding[:10]}")
    
    print("\n" + "=" * 60)
    print("How Analogies Work with Embeddings")
    print("=" * 60)
    
    print("\nAnalogies work through vector arithmetic:")
    print("  To solve: A is to B as C is to ?")
    print("  We calculate: vector(B) - vector(A) + vector(C)")
    print("  Then find the word whose vector is closest to this result")
    print("\nExample: man:woman::king:?")
    print("  vector(woman) - vector(man) + vector(king) ≈ vector(queen)")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
