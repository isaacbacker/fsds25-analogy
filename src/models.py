#!/usr/bin/env python3
"""
Model management module for word embeddings.

This module handles downloading and loading pre-trained word embedding models,
primarily using gensim's downloader API for convenience.
"""

import os
import ssl
import certifi
import urllib.request
import shutil

# Fix SSL certificate verification on macOS
# Use certifi's certificate bundle instead of system certificates
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Create SSL context with certifi certificates
_ssl_context = ssl.create_default_context(cafile=certifi.where())

# Patch urllib.request to use certifi certificates for SSL verification
# This fixes the SSL certificate verification issue on macOS
_original_urlopen = urllib.request.urlopen
_original_urlretrieve = urllib.request.urlretrieve

def _patched_urlopen(url, *args, **kwargs):
    """Patch urlopen to use certifi certificates."""
    # If it's HTTPS, add SSL context
    if isinstance(url, (str, bytes)) and (isinstance(url, str) and url.startswith('https://')):
        if 'context' not in kwargs:
            kwargs['context'] = _ssl_context
    elif isinstance(url, urllib.request.Request):
        if url.get_full_url().startswith('https://'):
            if 'context' not in kwargs:
                kwargs['context'] = _ssl_context
    return _original_urlopen(url, *args, **kwargs)

def _patched_urlretrieve(url, filename, reporthook=None, data=None):
    """Patch urlretrieve to use certifi certificates."""
    if isinstance(url, str) and url.startswith('https://'):
        # For HTTPS URLs, we need to use urlopen with SSL context
        response = _patched_urlopen(url, data=data)
        try:
            # Get file size if available
            file_size = None
            if hasattr(response, 'headers') and 'Content-Length' in response.headers:
                file_size = int(response.headers['Content-Length'])
            
            with open(filename, 'wb') as f:
                block_size = 8192
                bytes_read = 0
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)
                    if reporthook:
                        reporthook(bytes_read, file_size if file_size else -1, file_size if file_size else -1)
        finally:
            response.close()
        return (filename, response.headers if hasattr(response, 'headers') else {})
    else:
        # For non-HTTPS, use original function
        return _original_urlretrieve(url, filename, reporthook, data)

# Apply the patches
urllib.request.urlopen = _patched_urlopen
urllib.request.urlretrieve = _patched_urlretrieve

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
        self._ensure_gensim_cache()
    
    def _ensure_gensim_cache(self):
        """
        Ensure gensim's cache is initialized by loading model information.
        This will download information.json if it doesn't exist.
        """
        try:
            # Try to access the info, which will trigger cache initialization
            # This will automatically download information.json if missing and internet is available
            api.info()
        except (ValueError, FileNotFoundError):
            # If cache initialization fails, it will be handled at model load time
            # with a clearer error message
            pass
    
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
        
        # Ensure cache is initialized before loading
        # This will automatically download information.json if missing
        try:
            api.info()
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to initialize gensim cache. Please ensure you have internet "
                "connectivity. Gensim needs to download model information file. "
                "If the problem persists, try deleting ~/gensim-data and running again. "
                f"Original error: {e}"
            )
        
        # Download and load the model using gensim's downloader
        model = api.load(model_name)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {len(model.index_to_key):,} words")
        print(f"  Vector dimensions: {model.vector_size}")
        
        # Cache the model
        self._models[model_name] = model
        
        return model
    
    def load_fasttext_wiki_news(self):
        """
        Load fastText English word vectors (wiki-news-subwords-300) using gensim downloader.

        This model:
            - Is 300-dimensional
            - Uses subword information (handles OOV words better than word2vec)
            - Is a good modern successor to word2vec-google-news-300 for analogies.

        Returns:
            Loaded fastText model (KeyedVectors)
        """
        model_name = 'fasttext-wiki-news-subwords-300'

        if model_name in self._models:
            print(f"Using cached model: {model_name}")
            return self._models[model_name]

        print(f"Loading fastText wiki-news-subwords model...")
        print(f"Model: {model_name}")
        print("This may take a few minutes on first download...")

        # Ensure cache is initialized before loading
        try:
            api.info()
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to initialize gensim cache. Please ensure you have internet "
                "connectivity. Gensim needs to download model information file. "
                "If the problem persists, try deleting ~/gensim-data and running again. "
                f"Original error: {e}"
            )

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
        
        # Ensure cache is initialized before loading
        # This will automatically download information.json if missing
        try:
            api.info()
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to initialize gensim cache. Please ensure you have internet "
                "connectivity. Gensim needs to download model information file. "
                "If the problem persists, try deleting ~/gensim-data and running again. "
                f"Original error: {e}"
            )
        
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
        print("\nFastText models:")
        print("  - fasttext-wiki-news-subwords-300")
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
