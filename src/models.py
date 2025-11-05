#!/usr/bin/env python3
"""
Model management module for word embeddings.

This module handles downloading and loading pre-trained word embedding models,
primarily using gensim's downloader API for convenience.
"""

import os
import gensim.downloader as api
from gensim.models import KeyedVectors


class ModelManager:
    """Manages loading and caching of word embedding models."""
    
    def __init__(self, cache_dir='data/models'):
        """
        Initialize the ModelManager.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._models = {}
    
    def load_word2vec_google_news(self):
        """
        Load Google's pre-trained Word2Vec model using gensim downloader.
        
        This model was trained on Google News corpus (about 100 billion words).
        The model contains 300-dimensional vectors for 3 million words and phrases.
        
        Note: This is a large file (~1.6 GB download, ~3.5GB in memory).
        
        Returns:
            Loaded Word2Vec model (KeyedVectors)
        """
        model_name = 'word2vec-google-news-300'
        
        if model_name in self._models:
            print(f"Using cached model: {model_name}")
            return self._models[model_name]
        
        print(f"Loading Word2Vec Google News model...")
        print(f"Model: {model_name}")
        print(f"Size: ~1.6GB download, ~3.5GB in memory")
        print("This may take a few minutes on first download...")
        
        # Download and load the model using gensim's downloader
        model = api.load(model_name)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {len(model.index_to_key):,} words")
        print(f"  Vector dimensions: {model.vector_size}")
        
        # Cache the model
        self._models[model_name] = model
        
        return model
    
    def load_glove(self, dimension=100):
        """
        Load GloVe model using gensim downloader.
        
        Available dimensions via gensim: 25, 50, 100, 200
        Trained on Wikipedia 2014 + Gigaword 5 (6B tokens).
        
        Args:
            dimension: Dimension of the word vectors (25, 50, 100, or 200)
            
        Returns:
            Loaded GloVe model (KeyedVectors)
        """
        valid_dims = [25, 50, 100, 200]
        if dimension not in valid_dims:
            raise ValueError(f"Invalid dimension: {dimension}. Choose from {valid_dims}")
        
        model_name = f'glove-wiki-gigaword-{dimension}'
        
        if model_name in self._models:
            print(f"Using cached model: {model_name}")
            return self._models[model_name]
        
        print(f"Loading GloVe model...")
        print(f"Model: {model_name}")
        print(f"Dimension: {dimension}")
        print("This may take a few minutes on first download...")
        
        # Download and load the model using gensim's downloader
        model = api.load(model_name)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {len(model.index_to_key):,} words")
        print(f"  Vector dimensions: {model.vector_size}")
        
        # Cache the model
        self._models[model_name] = model
        
        return model
    
    def load_custom_model(self, filepath, binary=True):
        """
        Load a custom word2vec format model from disk.
        
        Args:
            filepath: Path to the model file
            binary: Whether the model is in binary format
            
        Returns:
            Loaded model (KeyedVectors)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        print(f"Loading custom model from {filepath}...")
        model = KeyedVectors.load_word2vec_format(filepath, binary=binary)
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {len(model.index_to_key):,} words")
        print(f"  Vector dimensions: {model.vector_size}")
        
        return model
    
    def list_available_models(self):
        """
        List all models available through gensim downloader.
        
        Returns:
            Dictionary of available models with their info
        """
        print("Available models through gensim downloader:")
        print("\nWord2Vec models:")
        print("  - word2vec-google-news-300")
        print("\nGloVe models:")
        print("  - glove-wiki-gigaword-25")
        print("  - glove-wiki-gigaword-50")
        print("  - glove-wiki-gigaword-100")
        print("  - glove-wiki-gigaword-200")
        print("  - glove-wiki-gigaword-300")
        print("  - glove-twitter-25")
        print("  - glove-twitter-50")
        print("  - glove-twitter-100")
        print("  - glove-twitter-200")
        
        # Get full info
        info = api.info()
        return info


def load_word2vec_model():
    """
    Convenience function to load the default Word2Vec model.
    
    Returns:
        Loaded Word2Vec model (KeyedVectors)
    """
    manager = ModelManager()
    return manager.load_word2vec_google_news()
