#!/usr/bin/env python3
"""
Main CLI entry point for Word2Vec analogy testing.

This script provides a command-line interface for testing word analogies
using pre-trained Word2Vec models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelManager, load_word2vec_model
from src.analogy_tests import (
    run_analogy_test_suite,
    print_test_summary,
    test_analogy,
    explore_nearest_neighbors,
    calculate_vector_arithmetic
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Test Word2Vec analogies and explore word embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default analogy test suite
  python analogy.py
  
  # Test a specific analogy
  python analogy.py --test man woman king queen
  
  # Explore nearest neighbors
  python analogy.py --neighbors king --top 20
  
  # Custom vector arithmetic
  python analogy.py --arithmetic --positive king woman --negative man
  
  # Use GloVe model instead
  python analogy.py --model glove --glove-dim 100
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        choices=['word2vec', 'glove'],
        default='word2vec',
        help='Which model to use (default: word2vec)'
    )
    parser.add_argument(
        '--glove-dim',
        type=int,
        choices=[25, 50, 100, 200],
        default=100,
        help='Dimension for GloVe model (default: 100)'
    )
    parser.add_argument(
        '--custom-model',
        type=str,
        help='Path to custom word2vec format model file'
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Custom model is in binary format'
    )
    
    # Testing modes
    parser.add_argument(
        '--test',
        nargs=4,
        metavar=('WORD_A', 'WORD_B', 'WORD_C', 'TARGET'),
        help='Test a specific analogy: A:B::C:TARGET'
    )
    parser.add_argument(
        '--suite',
        action='store_true',
        help='Run the full test suite (default if no other mode specified)'
    )
    parser.add_argument(
        '--neighbors',
        type=str,
        metavar='WORD',
        help='Explore nearest neighbors of a word'
    )
    parser.add_argument(
        '--arithmetic',
        action='store_true',
        help='Perform custom vector arithmetic'
    )
    
    # Vector arithmetic arguments
    parser.add_argument(
        '--positive',
        nargs='+',
        metavar='WORD',
        help='Words to add in vector arithmetic'
    )
    parser.add_argument(
        '--negative',
        nargs='+',
        metavar='WORD',
        help='Words to subtract in vector arithmetic'
    )
    
    # Display options
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top results to show (default: 10)'
    )
    parser.add_argument(
        '--search-space',
        type=int,
        default=50000,
        help='Number of most frequent words to search (default: 50000)'
    )
    
    args = parser.parse_args()
    
    # Load model
    print("="*70)
    print("Word2Vec Analogy Testing Tool")
    print("="*70)
    
    try:
        manager = ModelManager()
        
        if args.custom_model:
            print(f"\nLoading custom model from: {args.custom_model}")
            model = manager.load_custom_model(args.custom_model, binary=args.binary)
        elif args.model == 'glove':
            print(f"\nLoading GloVe model (dimension: {args.glove_dim})...")
            model = manager.load_glove(args.glove_dim)
        else:
            print("\nLoading Word2Vec model...")
            model = manager.load_word2vec_google_news()
        
        print("\n" + "="*70)
        
        # Execute requested operation
        if args.test:
            # Test specific analogy
            word_a, word_b, word_c, target = args.test
            test_analogy(
                model, word_a, word_b, word_c, target,
                top_n=args.top,
                search_space=args.search_space
            )
            
        elif args.neighbors:
            # Explore nearest neighbors
            explore_nearest_neighbors(model, args.neighbors, n=args.top)
            
        elif args.arithmetic:
            # Custom vector arithmetic
            if not args.positive and not args.negative:
                print("Error: --arithmetic requires --positive and/or --negative words")
                return 1
            
            positive = args.positive if args.positive else []
            negative = args.negative if args.negative else []
            calculate_vector_arithmetic(model, positive, negative, topn=args.top)
            
        else:
            # Default: run test suite
            results = run_analogy_test_suite(model)
            print_test_summary(results)
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
