"""
LLM-based analogy solver using OpenAI API.

This module provides functionality to solve word analogies using Large Language Models
like GPT-3.5 or GPT-4.
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI


class LLMAnalogySolver:
    """
    A class for solving word analogies using Large Language Models.
    
    This class uses OpenAI's API to solve analogy problems and can compare
    results with traditional embedding-based approaches.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLMAnalogySolver.
        
        Args:
            api_key: OpenAI API key. If None, will try to read from OPENAI_API_KEY
                    environment variable.
            model: The OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Either pass it as an argument "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def solve_analogy(self, word_a: str, word_b: str, word_c: str, num_answers: int = 5) -> List[str]:
        """
        Solve word analogy: word_a is to word_b as word_c is to ?
        
        Example: "man" is to "woman" as "king" is to "queen"
        
        Args:
            word_a: First word in the analogy
            word_b: Second word in the analogy
            word_c: Third word in the analogy
            num_answers: Number of possible answers to generate
            
        Returns:
            List of possible answer words
        """
        prompt = f"""Complete this word analogy by providing {num_answers} possible answers.

Analogy: "{word_a}" is to "{word_b}" as "{word_c}" is to ?

Provide your answers as a simple numbered list, one word per line.
Example format:
1. word1
2. word2
3. word3

Focus on the most logical and common relationship between the words."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves word analogies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Validate response structure
            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("No response from OpenAI API")
            
            answer_text = response.choices[0].message.content
            if answer_text is None:
                raise RuntimeError("Empty response from OpenAI API")
            
            # Parse the answers from the response
            answers = self._parse_answers(answer_text)
            return answers[:num_answers]
            
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
    
    def _parse_answers(self, text: str) -> List[str]:
        """
        Parse answers from the LLM response.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            List of extracted words
        """
        answers = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering and punctuation
            # Handle formats like "1. word", "1) word", "- word", etc.
            parts = line.split('. ', 1) if '. ' in line else line.split(') ', 1) if ') ' in line else [line]
            if len(parts) > 1:
                word = parts[1].strip()
            else:
                word = parts[0].strip()
            
            # Remove any remaining punctuation and take first word
            word = word.split()[0] if word.split() else word
            word = word.strip('.,!?;:"-\'')
            
            if word and word.isalpha():
                answers.append(word.lower())
        
        return answers
    
    def explain_analogy(self, word_a: str, word_b: str, word_c: str, word_d: str) -> str:
        """
        Get an explanation of how the analogy works.
        
        Args:
            word_a: First word in the analogy
            word_b: Second word in the analogy
            word_c: Third word in the analogy
            word_d: Proposed answer
            
        Returns:
            Explanation string
        """
        prompt = f"""Explain the word analogy: "{word_a}" is to "{word_b}" as "{word_c}" is to "{word_d}"

Provide a clear, concise explanation of:
1. The relationship between {word_a} and {word_b}
2. How that same relationship applies to {word_c} and {word_d}
3. Whether {word_d} is a good answer and why

Keep your explanation to 2-3 sentences."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains word analogies clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            # Validate response structure
            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("No response from OpenAI API")
            
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from OpenAI API")
            
            return content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
    
    def compare_analogies(self, analogies: List[tuple]) -> Dict:
        """
        Compare multiple analogy solutions.
        
        Args:
            analogies: List of tuples (word_a, word_b, word_c, expected_answer)
            
        Returns:
            Dictionary with results for each analogy
        """
        results = {}
        
        for word_a, word_b, word_c, expected in analogies:
            answers = self.solve_analogy(word_a, word_b, word_c, num_answers=5)
            
            results[f"{word_a}:{word_b}::{word_c}:?"] = {
                "expected": expected,
                "predictions": answers,
                "correct": expected.lower() in [a.lower() for a in answers]
            }
        
        return results
