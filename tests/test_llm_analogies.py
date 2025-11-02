"""
Tests for the LLM analogies module.

Note: These tests require OPENAI_API_KEY to be set.
They are marked with pytest.mark.skipif to skip when the key is not available.
"""

import os
import pytest
from analogy.llm_analogies import LLMAnalogySolver


# Skip tests if API key is not available
skip_if_no_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


def test_initialization_without_key():
    """Test that initialization fails without API key."""
    # Temporarily remove the key
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    
    try:
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            LLMAnalogySolver()
    finally:
        # Restore the key if it existed
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


@skip_if_no_api_key
def test_initialization_with_key():
    """Test that initialization works with API key."""
    llm = LLMAnalogySolver()
    assert llm.client is not None
    assert llm.model == "gpt-3.5-turbo"


@skip_if_no_api_key
def test_initialization_custom_model():
    """Test initialization with custom model."""
    llm = LLMAnalogySolver(model="gpt-4")
    assert llm.model == "gpt-4"


@skip_if_no_api_key
def test_solve_analogy():
    """Test solving a basic analogy."""
    llm = LLMAnalogySolver()
    
    # Test classic analogy: man:woman::king:?
    answers = llm.solve_analogy("man", "woman", "king", num_answers=5)
    
    assert isinstance(answers, list)
    assert len(answers) <= 5
    assert all(isinstance(word, str) for word in answers)
    
    # Check that "queen" is likely in the answers
    answers_lower = [a.lower() for a in answers]
    assert "queen" in answers_lower


@skip_if_no_api_key
def test_solve_analogy_different_count():
    """Test solving analogy with different number of answers."""
    llm = LLMAnalogySolver()
    
    answers = llm.solve_analogy("paris", "france", "london", num_answers=3)
    
    assert len(answers) <= 3


@skip_if_no_api_key
def test_explain_analogy():
    """Test getting an explanation for an analogy."""
    llm = LLMAnalogySolver()
    
    explanation = llm.explain_analogy("man", "woman", "king", "queen")
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    # Should contain relevant words
    explanation_lower = explanation.lower()
    assert any(word in explanation_lower for word in ["man", "woman", "king", "queen", "gender", "relationship"])


@skip_if_no_api_key
def test_compare_analogies():
    """Test comparing multiple analogies."""
    llm = LLMAnalogySolver()
    
    analogies = [
        ("man", "woman", "king", "queen"),
        ("paris", "france", "london", "england"),
    ]
    
    results = llm.compare_analogies(analogies)
    
    assert isinstance(results, dict)
    assert len(results) == 2
    
    for key, value in results.items():
        assert "expected" in value
        assert "predictions" in value
        assert "correct" in value
        assert isinstance(value["correct"], bool)


def test_parse_answers():
    """Test the answer parsing method."""
    llm = LLMAnalogySolver(api_key="dummy_key_for_testing")
    
    # Test various formats
    text1 = "1. queen\n2. monarch\n3. ruler"
    answers1 = llm._parse_answers(text1)
    assert "queen" in answers1
    assert "monarch" in answers1
    
    text2 = "1) queen\n2) monarch\n3) ruler"
    answers2 = llm._parse_answers(text2)
    assert "queen" in answers2
    
    text3 = "queen\nmonarch\nruler"
    answers3 = llm._parse_answers(text3)
    assert "queen" in answers3


@skip_if_no_api_key
def test_geographic_analogy():
    """Test a geographic analogy."""
    llm = LLMAnalogySolver()
    
    answers = llm.solve_analogy("tokyo", "japan", "paris", num_answers=5)
    
    answers_lower = [a.lower() for a in answers]
    # Should find France
    assert "france" in answers_lower


@skip_if_no_api_key
def test_verb_tense_analogy():
    """Test a verb tense analogy."""
    llm = LLMAnalogySolver()
    
    answers = llm.solve_analogy("walk", "walked", "run", num_answers=5)
    
    answers_lower = [a.lower() for a in answers]
    # Should find "ran"
    assert "ran" in answers_lower
