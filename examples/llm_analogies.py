"""
Example demonstrating word analogies using LLM (OpenAI API).

This script shows how to:
1. Use OpenAI's API to solve word analogies
2. Compare LLM results with embedding-based results
3. Get explanations for analogies

Note: You need to set OPENAI_API_KEY environment variable or create a .env file
"""

import os
from dotenv import load_dotenv
from analogy import LLMAnalogySolver

# Load environment variables from .env file if it exists
load_dotenv()


def main():
    print("=" * 60)
    print("Word Analogies with LLM (OpenAI) Example")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not found!")
        print("\nTo use this example, you need to:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   OR create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print("\nExiting...")
        return
    
    try:
        # Initialize the LLM solver
        llm = LLMAnalogySolver(model="gpt-3.5-turbo")
        
        print("\n" + "=" * 60)
        print("Example 1: Classic Gender Analogy")
        print("=" * 60)
        
        # Classic analogy: man is to woman as king is to ?
        print("\nSolving: 'man' is to 'woman' as 'king' is to ?")
        answers = llm.solve_analogy("man", "woman", "king", num_answers=5)
        
        print("\nLLM Answers:")
        for i, word in enumerate(answers, 1):
            print(f"{i}. {word}")
        
        # Get explanation
        print("\nExplanation:")
        explanation = llm.explain_analogy("man", "woman", "king", "queen")
        print(explanation)
        
        print("\n" + "=" * 60)
        print("Example 2: Geographic Analogy")
        print("=" * 60)
        
        # Geographic analogy: paris is to france as london is to ?
        print("\nSolving: 'paris' is to 'france' as 'london' is to ?")
        answers = llm.solve_analogy("paris", "france", "london", num_answers=5)
        
        print("\nLLM Answers:")
        for i, word in enumerate(answers, 1):
            print(f"{i}. {word}")
        
        print("\n" + "=" * 60)
        print("Example 3: Comparative Adjective Analogy")
        print("=" * 60)
        
        # Adjective analogy: good is to better as bad is to ?
        print("\nSolving: 'good' is to 'better' as 'bad' is to ?")
        answers = llm.solve_analogy("good", "better", "bad", num_answers=5)
        
        print("\nLLM Answers:")
        for i, word in enumerate(answers, 1):
            print(f"{i}. {word}")
        
        print("\n" + "=" * 60)
        print("Example 4: Batch Analogy Testing")
        print("=" * 60)
        
        # Test multiple analogies
        test_analogies = [
            ("man", "woman", "king", "queen"),
            ("paris", "france", "london", "england"),
            ("cat", "kitten", "dog", "puppy"),
            ("swim", "swimming", "run", "running"),
        ]
        
        print("\nTesting multiple analogies:")
        results = llm.compare_analogies(test_analogies)
        
        for analogy, result in results.items():
            status = "✓" if result["correct"] else "✗"
            print(f"\n{status} {analogy}")
            print(f"   Expected: {result['expected']}")
            print(f"   Got: {', '.join(result['predictions'][:3])}")
        
        print("\n" + "=" * 60)
        print("Understanding LLM-based Analogies")
        print("=" * 60)
        
        print("\nHow LLMs solve analogies:")
        print("- LLMs use contextual understanding from training data")
        print("- They recognize patterns and relationships between words")
        print("- They can explain their reasoning in natural language")
        print("- Results may vary based on the model and temperature setting")
        
        print("\nDifferences from embedding-based approaches:")
        print("- Embeddings use fixed vector representations")
        print("- LLMs can understand context and nuance better")
        print("- LLMs can handle more complex relationships")
        print("- Embeddings are deterministic; LLMs can be creative")
        
        print("\n" + "=" * 60)
        print("Example Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check your API key and internet connection.")


if __name__ == "__main__":
    main()
