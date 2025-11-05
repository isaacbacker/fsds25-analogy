# fsds25-analogy

Repository for considering analogies using language models (Word2Vec and GloVe).

## Overview

This project provides tools and datasets for exploring and evaluating word analogies using pre-trained word embedding models. Word analogies follow the pattern: "A is to B as C is to D" (e.g., "man is to woman as king is to queen").

## Project Structure

```
fsds25-analogy/
├── analogy.py          # Main CLI entry point for testing analogies
├── src/                # Source code modules
│   ├── __init__.py     # Package initialization
│   ├── models.py       # Model loading and management
│   ├── analogy_tests.py # Analogy testing functions
│   └── utils.py        # Utility functions (download, extract)
├── data/               # Data files and cached models
│   ├── analogies.csv   # Standard word analogies dataset
│   └── models/         # Downloaded word embedding models (auto-created)
├── output/             # Output files from analysis
├── figures/            # Generated figures and visualizations
├── setup.sh            # Setup script (creates venv, installs deps)
├── requirements.txt    # Python dependencies
├── download_models.py  # (Legacy) Manual model downloader
├── word2vec_analogy.py # (Legacy) Original analogy tester
└── README.md           # This file
```

## Setup

### Quick Setup (Recommended)

Use the provided setup script to automatically create a virtual environment and install dependencies:

```bash
./setup.sh
```

Then activate the virtual environment:
```bash
source venv/bin/activate
```

### Manual Setup

Alternatively, set up manually:

#### 1. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 2. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Model Downloads

Models are automatically downloaded on first use via gensim's downloader. No manual download required!

The first time you run the analogy tool, it will download the Word2Vec model (~1.6 GB) automatically.

## Available Models

### Word2Vec
- **Model:** GoogleNews-vectors-negative300
- **Size:** ~1.5 GB
- **Dimensions:** 300
- **Vocabulary:** 3 million words and phrases
- **Training Data:** Google News corpus (~100 billion words)

### GloVe
- **Model:** glove.6B (Wikipedia 2014 + Gigaword 5)
- **Size:** ~800 MB (for all dimensions)
- **Dimensions:** 50, 100, 200, or 300
- **Vocabulary:** 400K words
- **Training Data:** 6 billion tokens

## Dataset

The `data/analogies.csv` file contains standard word analogies organized by category:

- **Gender:** man/woman, king/queen relationships
- **Capital-Country:** Paris/France, London/England relationships
- **Grammar:** verb forms, adjective comparatives
- **Animal-Young:** dog/puppy, cat/kitten relationships
- **Math:** mathematical operations and concepts
- **Ordinal:** number ordering relationships

Format: `word1,word2,word3,word4,category`

Example: `man,woman,king,queen,gender`

## Usage

### Quick Start

Run the default analogy test suite:

```bash
python analogy.py
```

This will load the Word2Vec model and test several classic analogies like "man:woman::king:queen".

### CLI Options

```bash
# Test a specific analogy
python analogy.py --test man woman king queen

# Explore nearest neighbors of a word
python analogy.py --neighbors king --top 20

# Custom vector arithmetic
python analogy.py --arithmetic --positive king woman --negative man

# Use GloVe model instead of Word2Vec
python analogy.py --model glove --glove-dim 100

# Use a custom model file
python analogy.py --custom-model path/to/model.bin --binary

# Show more results
python analogy.py --top 20 --search-space 100000
```

### Python API Usage

You can also use the modules directly in your Python code:

```python
from src.models import ModelManager
from src.analogy_tests import test_analogy, run_analogy_test_suite, print_test_summary

# Load a model
manager = ModelManager()
model = manager.load_word2vec_google_news()

# Test a single analogy
test_analogy(model, "man", "woman", "king", "queen")

# Run full test suite
results = run_analogy_test_suite(model)
print_test_summary(results)

# Explore nearest neighbors
from src.analogy_tests import explore_nearest_neighbors
explore_nearest_neighbors(model, "king", n=10)
```

## Dependencies

- **numpy** (>=1.21.0): Numerical computing
- **pandas** (>=1.3.0): Data manipulation and analysis
- **gensim** (>=4.0.0): Word embedding models and similarity operations
- **requests** (>=2.26.0): HTTP library for downloads
- **tqdm** (>=4.62.0): Progress bars
- **scikit-learn** (>=1.0.0): Machine learning utilities
- **matplotlib** (>=3.4.0): Plotting and visualization
- **scipy** (>=1.7.0): Scientific computing (for distance calculations)

## Legacy Files

The project has been refactored for better organization. The following files are kept for reference but are no longer the primary entry points:

- `download_models.py`: Original manual model downloader (models now auto-download via gensim)
- `word2vec_analogy.py`: Original analogy testing script (replaced by `analogy.py` with modular `src/` package)

You can still use these files if needed, but `analogy.py` is now the recommended entry point.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## References

- Word2Vec: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- GloVe: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- Gensim: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
