"""
Word Analogies Project using Embeddings and LLMs

This package provides tools for exploring word analogies using pre-trained
word embeddings (via gensim) and LLM APIs.
"""

__version__ = "0.1.0"

from .embeddings import WordEmbeddings
from .llm_analogies import LLMAnalogySolver

__all__ = ["WordEmbeddings", "LLMAnalogySolver"]
