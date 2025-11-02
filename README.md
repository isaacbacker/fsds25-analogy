# fsds25-analogy

Repository for exploring word analogies using language models and embeddings.

## Overview

This project demonstrates the basics of word analogies using two approaches:
1. **Word Embeddings** (Word2Vec, GloVe) via gensim
2. **Large Language Models** (GPT-3.5, GPT-4) via OpenAI API

The classic example is: **"man" : "woman" :: "king" : "queen"**

This means "man is to woman as king is to queen" - both pairs share a similar relationship (gender).

## What are Word Embeddings?

Word embeddings are dense vector representations of words where:
- Each word is represented as a vector of numbers (e.g., 100-300 dimensions)
- Words with similar meanings have similar vectors
- Mathematical operations on vectors can capture semantic relationships

For example:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

## Features

- 📊 **Embedding-based analogies** using pre-trained Word2Vec and GloVe models
- 🤖 **LLM-based analogies** using OpenAI's GPT models
- 🔍 **Word similarity** calculations
- 📖 **Analogy explanations** to understand relationships
- 🧪 **Comparison tools** to evaluate different approaches

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/oxfordinternetinstitute/fsds25-analogy.git
cd fsds25-analogy

# Install dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

### OpenAI API Setup (Optional)

To use LLM features, you need an OpenAI API key:

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set up your environment:

```bash
# Option 1: Export as environment variable
export OPENAI_API_KEY='your-api-key-here'

# Option 2: Create a .env file
cp .env.example .env
# Edit .env and add your API key
```

## Quick Start

### Using Word Embeddings

```python
from analogy import WordEmbeddings

# Initialize with a model
embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-100")

# Download the model (cached after first download)
embeddings.download_model()

# Solve an analogy: man:woman::king:?
results = embeddings.solve_analogy("man", "woman", "king", topn=5)
print(results)
# Output: [('queen', 0.8523), ('monarch', 0.7234), ...]

# Find similar words
similar = embeddings.most_similar("computer", topn=5)
print(similar)
# Output: [('computers', 0.89), ('laptop', 0.78), ...]

# Calculate word similarity
similarity = embeddings.similarity("king", "queen")
print(f"Similarity: {similarity:.4f}")
# Output: Similarity: 0.7845
```

### Using LLM (OpenAI)

```python
from analogy import LLMAnalogySolver

# Initialize (requires OPENAI_API_KEY)
llm = LLMAnalogySolver(model="gpt-3.5-turbo")

# Solve an analogy
answers = llm.solve_analogy("man", "woman", "king", num_answers=5)
print(answers)
# Output: ['queen', 'monarch', 'ruler', 'sovereign', 'prince']

# Get an explanation
explanation = llm.explain_analogy("man", "woman", "king", "queen")
print(explanation)
# Output: "The relationship between 'man' and 'woman' is one of gender..."
```

## Examples

The `examples/` directory contains full working examples:

### 1. Basic Embeddings Example
```bash
python examples/basic_embeddings.py
```

Demonstrates:
- Downloading and using pre-trained models
- Solving classic analogies (gender, geography, verb tense)
- Word similarity calculations
- Understanding embeddings

### 2. LLM Analogies Example
```bash
python examples/llm_analogies.py
```

Demonstrates:
- Using OpenAI's API for analogies
- Getting explanations
- Batch testing multiple analogies

### 3. Comparison Example
```bash
python examples/compare_methods.py
```

Demonstrates:
- Side-by-side comparison of both approaches
- Strengths and weaknesses of each method
- When to use which approach

## Project Structure

```
fsds25-analogy/
├── src/analogy/           # Main package
│   ├── __init__.py       # Package initialization
│   ├── embeddings.py     # Embedding-based analogies
│   └── llm_analogies.py  # LLM-based analogies
├── examples/             # Example scripts
│   ├── basic_embeddings.py
│   ├── llm_analogies.py
│   └── compare_methods.py
├── tests/                # Test suite
│   ├── test_embeddings.py
│   └── test_llm_analogies.py
├── requirements.txt      # Dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/analogy

# Run only embedding tests
pytest tests/test_embeddings.py

# Run only LLM tests (requires OPENAI_API_KEY)
pytest tests/test_llm_analogies.py
```

**Note:** Some LLM tests will be skipped if `OPENAI_API_KEY` is not set.

## Available Models

### Embedding Models (via gensim)

Fast, offline, and free:
- `glove-wiki-gigaword-50` - Small, fast (65MB)
- `glove-wiki-gigaword-100` - Medium, balanced (128MB) ⭐ Recommended
- `glove-wiki-gigaword-200` - Large, better quality (252MB)
- `glove-wiki-gigaword-300` - Largest, best quality (376MB)
- `word2vec-google-news-300` - Google News trained (1.6GB)
- `fasttext-wiki-news-subwords-300` - FastText model (958MB)

### LLM Models (via OpenAI)

Requires API key and internet:
- `gpt-3.5-turbo` - Fast, cost-effective ⭐ Recommended
- `gpt-4` - More capable, higher cost
- `gpt-4-turbo` - Balance of capability and speed

## How Analogies Work

### Embedding Approach

Word embeddings solve analogies using **vector arithmetic**:

```
king - man + woman ≈ queen
```

This works because:
1. Each word is a vector in high-dimensional space
2. Relationships are captured as vector offsets
3. Similar relationships have similar vector offsets

### LLM Approach

LLMs solve analogies using:
1. **Pattern recognition** from training data
2. **Contextual understanding** of word relationships
3. **Semantic reasoning** about language

LLMs can also explain *why* an analogy works, which embeddings cannot.

## Common Analogies

Here are some classic analogies to try:

**Gender:**
- man:woman::king:queen
- boy:girl::brother:sister
- actor:actress::waiter:waitress

**Geography:**
- paris:france::london:england
- tokyo:japan::berlin:germany
- madrid:spain::rome:italy

**Verb Tense:**
- walk:walked::run:ran
- sing:sang::swim:swam
- go:went::do:did

**Comparatives:**
- good:better::bad:worse
- big:bigger::small:smaller
- fast:faster::slow:slower

**Part-Whole:**
- finger:hand::toe:foot
- page:book::scene:movie
- wheel:car::wing:airplane

## Learning Resources

To learn more about embeddings and analogies:

1. **Word2Vec Paper:** [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
2. **GloVe Website:** [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/)
3. **Gensim Documentation:** [gensim.models](https://radimrehurek.com/gensim/models/word2vec.html)
4. **Illustrated Word2Vec:** [Jay Alammar's Blog](https://jalammar.github.io/illustrated-word2vec/)

## Troubleshooting

### Model Download Issues

If model downloads fail:
```python
# Try a smaller model first
embeddings = WordEmbeddings(model_name="glove-wiki-gigaword-50")

# Check your internet connection
# Models are cached in ~/gensim-data/ after first download
```

### OpenAI API Issues

If LLM calls fail:
```bash
# Check your API key
echo $OPENAI_API_KEY

# Verify key is valid at https://platform.openai.com/api-keys

# Check API usage limits and billing
```

### Out of Memory

If you get memory errors with large models:
- Use smaller models (glove-50 or glove-100)
- Close other applications
- Use a machine with more RAM

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Pre-trained models from [gensim-data](https://github.com/RaRe-Technologies/gensim-data)
- OpenAI for GPT models
- Stanford NLP for GloVe embeddings
- Google for Word2Vec models

## Contact

For questions or issues, please open an issue on GitHub.
