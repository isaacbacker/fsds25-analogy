"""
Simple script to test the basic functionality without full model downloads.
"""

from analogy import WordEmbeddings, LLMAnalogySolver
import sys

def test_embeddings_basic():
    """Test basic embedding functionality."""
    print("=" * 60)
    print("Testing WordEmbeddings class")
    print("=" * 60)
    
    # Test initialization
    print("\n1. Testing initialization...")
    embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-50")
    print("   ✓ Initialization successful")
    
    # Test model listing
    print("\n2. Testing model listing...")
    models = embeddings.list_available_models()
    print(f"   ✓ Found {len(models)} available models")
    print(f"   ✓ Sample models: {models[:3]}")
    
    # Test that model is not loaded yet
    print("\n3. Testing model state...")
    assert embeddings.model is None, "Model should not be loaded yet"
    print("   ✓ Model correctly not loaded initially")
    
    # Test error handling without loaded model
    print("\n4. Testing error handling...")
    try:
        embeddings.get_embedding("test")
        print("   ✗ Should have raised an error")
        return False
    except ValueError as e:
        if "Model not loaded" in str(e):
            print("   ✓ Correct error raised for unloaded model")
        else:
            print(f"   ✗ Wrong error: {e}")
            return False
    
    print("\n✓ All WordEmbeddings basic tests passed!")
    return True


def test_llm_basic():
    """Test basic LLM functionality."""
    print("\n" + "=" * 60)
    print("Testing LLMAnalogySolver class")
    print("=" * 60)
    
    # Test initialization without API key
    print("\n1. Testing initialization without API key...")
    try:
        # Temporarily save and remove API key
        import os
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            llm = LLMAnalogySolver()
            print("   ✗ Should have raised an error")
            return False
        except ValueError as e:
            if "API key" in str(e):
                print("   ✓ Correct error raised for missing API key")
            else:
                print(f"   ✗ Wrong error: {e}")
                return False
        finally:
            # Restore API key if it existed
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False
    
    # Test initialization with dummy key
    print("\n2. Testing initialization with dummy key...")
    try:
        llm = LLMAnalogySolver(api_key="dummy_key")
        print("   ✓ Initialization successful with API key")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test answer parsing
    print("\n3. Testing answer parsing...")
    test_text = "1. queen\n2. monarch\n3. ruler"
    answers = llm._parse_answers(test_text)
    if "queen" in answers and "monarch" in answers:
        print(f"   ✓ Answer parsing works: {answers}")
    else:
        print(f"   ✗ Answer parsing failed: {answers}")
        return False
    
    print("\n✓ All LLMAnalogySolver basic tests passed!")
    return True


def main():
    """Run all basic tests."""
    print("\n" + "=" * 60)
    print("Basic Functionality Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test embeddings
    try:
        results.append(("WordEmbeddings", test_embeddings_basic()))
    except Exception as e:
        print(f"\n✗ WordEmbeddings tests failed with exception: {e}")
        results.append(("WordEmbeddings", False))
    
    # Test LLM
    try:
        results.append(("LLMAnalogySolver", test_llm_basic()))
    except Exception as e:
        print(f"\n✗ LLMAnalogySolver tests failed with exception: {e}")
        results.append(("LLMAnalogySolver", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All basic tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
