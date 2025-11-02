"""
Tests for the word embeddings module.
"""

import pytest
import numpy as np
from analogy.embeddings import WordEmbeddings


# Use a small model for faster testing
TEST_MODEL = "glove-wiki-gigaword-50"


@pytest.fixture(scope="module")
def embeddings():
    """Create an embeddings instance with a small model for testing."""
    emb = WordEmbeddings(model_name=TEST_MODEL)
    emb.download_model()
    return emb


def test_model_initialization():
    """Test that the model can be initialized."""
    emb = WordEmbeddings(model_name=TEST_MODEL)
    assert emb.model_name == TEST_MODEL
    assert emb.model is None


def test_model_download(embeddings):
    """Test that the model downloads and loads correctly."""
    assert embeddings.model is not None
    assert len(embeddings.model) > 0


def test_get_embedding(embeddings):
    """Test getting word embeddings."""
    # Test a common word
    vec = embeddings.get_embedding("king")
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 0
    
    # Test a word not in vocabulary
    vec = embeddings.get_embedding("xyzabc123notaword")
    assert vec is None


def test_solve_analogy(embeddings):
    """Test solving word analogies."""
    # Classic analogy: man:woman::king:?
    results = embeddings.solve_analogy("man", "woman", "king", topn=5)
    
    assert len(results) <= 5
    assert all(isinstance(r, tuple) for r in results)
    assert all(len(r) == 2 for r in results)
    
    # Check that "queen" is in top results
    words = [word for word, score in results]
    assert "queen" in words or "queens" in words


def test_solve_analogy_invalid_word(embeddings):
    """Test that solving analogy with invalid word raises error."""
    with pytest.raises(ValueError):
        embeddings.solve_analogy("xyznotaword", "woman", "king")


def test_similarity(embeddings):
    """Test word similarity calculation."""
    # Similar words should have high similarity
    sim = embeddings.similarity("king", "queen")
    assert 0 < sim < 1
    assert sim > 0.5  # Should be fairly similar
    
    # Dissimilar words should have lower similarity
    sim = embeddings.similarity("king", "computer")
    assert sim < 0.7


def test_similarity_invalid_word(embeddings):
    """Test that similarity with invalid word raises error."""
    with pytest.raises(ValueError):
        embeddings.similarity("xyznotaword", "king")


def test_most_similar(embeddings):
    """Test finding most similar words."""
    results = embeddings.most_similar("computer", topn=5)
    
    assert len(results) <= 5
    assert all(isinstance(r, tuple) for r in results)
    
    # Check that similar words are returned
    words = [word for word, score in results]
    # Should contain technology-related words
    assert any(word in ["computers", "software", "technology", "pc"] for word in words)


def test_most_similar_invalid_word(embeddings):
    """Test that most_similar with invalid word raises error."""
    with pytest.raises(ValueError):
        embeddings.most_similar("xyznotaword")


def test_analogy_explanation(embeddings):
    """Test analogy explanation."""
    explanation = embeddings.analogy_explanation("man", "woman", "king", "queen")
    
    assert isinstance(explanation, dict)
    assert "analogy" in explanation
    assert "relationship_similarity" in explanation
    assert "predicted_similarity" in explanation
    assert "explanation" in explanation
    
    # Check values are reasonable
    assert -1 <= explanation["relationship_similarity"] <= 1
    assert -1 <= explanation["predicted_similarity"] <= 1


def test_analogy_explanation_invalid_word(embeddings):
    """Test that analogy explanation with invalid word raises error."""
    with pytest.raises(ValueError):
        embeddings.analogy_explanation("xyznotaword", "woman", "king", "queen")


def test_model_not_loaded_error():
    """Test that operations fail when model is not loaded."""
    emb = WordEmbeddings(model_name=TEST_MODEL)
    
    with pytest.raises(ValueError, match="Model not loaded"):
        emb.get_embedding("king")
    
    with pytest.raises(ValueError, match="Model not loaded"):
        emb.solve_analogy("man", "woman", "king")
    
    with pytest.raises(ValueError, match="Model not loaded"):
        emb.similarity("king", "queen")
    
    with pytest.raises(ValueError, match="Model not loaded"):
        emb.most_similar("king")


def test_geographic_analogy(embeddings):
    """Test a geographic analogy."""
    # Paris:France::London:?
    results = embeddings.solve_analogy("paris", "france", "london", topn=5)
    words = [word.lower() for word, score in results]
    
    # Should find Britain, England, or UK
    assert any(word in ["england", "britain", "uk"] for word in words)


def test_list_available_models(embeddings):
    """Test listing available models."""
    models = embeddings.list_available_models()
    
    assert isinstance(models, list)
    assert len(models) > 0
    assert TEST_MODEL in models
