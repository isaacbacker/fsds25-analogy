"""
Word embeddings module using gensim for word analogies.

This module provides functionality to download and use pre-trained word embeddings
to solve word analogy problems like "man":"woman"::"king":"queen".
"""

import gensim.downloader as api
import numpy as np
from typing import List, Tuple, Optional


class WordEmbeddings:
    """
    A class for working with word embeddings and solving word analogies.
    
    This class uses gensim to download and load pre-trained word embedding models
    and provides methods to solve analogy problems.
    """
    
    def __init__(self, model_name: str = "word2vec-google-news-300"):
        """
        Initialize the WordEmbeddings class.
        
        Args:
            model_name: Name of the pre-trained model to use.
                       Default is 'word2vec-google-news-300'.
                       Other options include:
                       - 'glove-wiki-gigaword-100'
                       - 'glove-wiki-gigaword-200'
                       - 'glove-wiki-gigaword-300'
                       - 'fasttext-wiki-news-subwords-300'
        """
        self.model_name = model_name
        self.model = None
        
    def download_model(self) -> None:
        """
        Download and load the pre-trained word embedding model.
        
        This may take some time depending on the model size and internet speed.
        Models are cached locally after the first download.
        """
        print(f"Downloading model: {self.model_name}")
        print("This may take several minutes on first run...")
        self.model = api.load(self.model_name)
        print(f"Model loaded successfully. Vocabulary size: {len(self.model)}")
        
    def list_available_models(self) -> List[str]:
        """
        List all available pre-trained models from gensim.
        
        Returns:
            List of model names that can be downloaded.
        """
        return list(api.info()['models'].keys())
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.
        
        Args:
            word: The word to get the embedding for.
            
        Returns:
            numpy array of the word embedding, or None if word not in vocabulary.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")
        
        try:
            return self.model[word]
        except KeyError:
            return None
    
    def solve_analogy(self, word_a: str, word_b: str, word_c: str, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Solve word analogy: word_a is to word_b as word_c is to ?
        
        This uses vector arithmetic: word_b - word_a + word_c
        Example: "man" is to "woman" as "king" is to "queen"
        
        Args:
            word_a: First word in the analogy (e.g., "man")
            word_b: Second word in the analogy (e.g., "woman")
            word_c: Third word in the analogy (e.g., "king")
            topn: Number of most similar words to return
            
        Returns:
            List of tuples (word, similarity_score) representing possible answers.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")
        
        try:
            # Use gensim's built-in analogy solver
            results = self.model.most_similar(positive=[word_b, word_c], negative=[word_a], topn=topn)
            return results
        except KeyError as e:
            raise ValueError(f"One or more words not in vocabulary: {e}")
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")
        
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            raise ValueError(f"One or more words not in vocabulary: {e}")
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find the most similar words to the given word.
        
        Args:
            word: The word to find similar words for
            topn: Number of similar words to return
            
        Returns:
            List of tuples (word, similarity_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")
        
        try:
            return self.model.most_similar(word, topn=topn)
        except KeyError:
            raise ValueError(f"Word '{word}' not in vocabulary")
    
    def analogy_explanation(self, word_a: str, word_b: str, word_c: str, word_d: str) -> dict:
        """
        Explain how well word_d completes the analogy word_a:word_b::word_c:word_d
        
        Args:
            word_a: First word in the analogy
            word_b: Second word in the analogy
            word_c: Third word in the analogy
            word_d: Proposed answer
            
        Returns:
            Dictionary with analysis of the analogy quality
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")
        
        # Get embeddings
        vec_a = self.get_embedding(word_a)
        vec_b = self.get_embedding(word_b)
        vec_c = self.get_embedding(word_c)
        vec_d = self.get_embedding(word_d)
        
        if any(v is None for v in [vec_a, vec_b, vec_c, vec_d]):
            raise ValueError("One or more words not in vocabulary")
        
        # Calculate relationship vectors
        relationship_ab = vec_b - vec_a
        relationship_cd = vec_d - vec_c
        
        # Calculate similarity between relationships
        norm_ab = np.linalg.norm(relationship_ab)
        norm_cd = np.linalg.norm(relationship_cd)
        
        if norm_ab == 0 or norm_cd == 0:
            raise ValueError("Words are identical in one or both pairs, cannot compute relationship")
        
        relationship_similarity = np.dot(relationship_ab, relationship_cd) / (norm_ab * norm_cd)
        
        # Calculate how close word_d is to the ideal answer
        ideal_d = vec_b - vec_a + vec_c
        norm_d = np.linalg.norm(vec_d)
        norm_ideal = np.linalg.norm(ideal_d)
        
        if norm_d == 0 or norm_ideal == 0:
            raise ValueError("Cannot compute similarity with zero vector")
        
        predicted_similarity = np.dot(vec_d, ideal_d) / (norm_d * norm_ideal)
        
        return {
            "analogy": f"{word_a}:{word_b}::{word_c}:{word_d}",
            "relationship_similarity": float(relationship_similarity),
            "predicted_similarity": float(predicted_similarity),
            "explanation": f"The relationship '{word_a}→{word_b}' is {relationship_similarity:.2%} similar to '{word_c}→{word_d}'"
        }
