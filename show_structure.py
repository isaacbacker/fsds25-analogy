"""
Project structure visualization for fsds25-analogy
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     FSDS25-ANALOGY PROJECT STRUCTURE                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

📦 fsds25-analogy/
┃
┣━━ 🎯 analogy.py                 Main entry point (CLI)
┃                                  - Argument parsing
┃                                  - Coordinates modules
┃                                  - User interface
┃
┣━━ 📂 src/                       Source code package
┃   ┃
┃   ┣━━ __init__.py               Package initialization
┃   ┃
┃   ┣━━ 🔧 models.py              Model management
┃   ┃                             - ModelManager class
┃   ┃                             - load_word2vec_google_news()
┃   ┃                             - load_glove()
┃   ┃                             - load_custom_model()
┃   ┃
┃   ┣━━ 🧪 analogy_tests.py       Analogy testing
┃   ┃                             - test_analogy()
┃   ┃                             - run_analogy_test_suite()
┃   ┃                             - print_test_summary()
┃   ┃                             - explore_nearest_neighbors()
┃   ┃                             - calculate_vector_arithmetic()
┃   ┃
┃   └━━ 🛠️  utils.py              Utilities (legacy support)
┃                                 - download_file()
┃                                 - extract_zip()
┃                                 - extract_gzip()
┃
┣━━ 📂 data/                      Data files
┃   ┣━━ analogies.csv             Standard analogies dataset
┃   └━━ models/                   Cached models (auto-created)
┃
┣━━ 📂 output/                    Analysis outputs
┣━━ 📂 figures/                   Visualizations
┃
┣━━ 🚀 setup.sh                   One-command setup script
┣━━ 📋 requirements.txt           Python dependencies
┣━━ 📖 README.md                  Project documentation
┣━━ 📝 REFACTORING.md             Refactoring guide
┃
┗━━ 🗄️  Legacy files (reference only)
    ┣━━ download_models.py
    └━━ word2vec_analogy.py

╔═══════════════════════════════════════════════════════════════════════════╗
║                              USAGE EXAMPLES                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

▶ Setup
  ./setup.sh && source venv/bin/activate

▶ Run default test suite
  python analogy.py

▶ Test specific analogy
  python analogy.py --test man woman king queen

▶ Explore word neighbors
  python analogy.py --neighbors king --top 20

▶ Vector arithmetic
  python analogy.py --arithmetic --positive king woman --negative man

▶ Use GloVe model
  python analogy.py --model glove --glove-dim 100

▶ Show help
  python analogy.py --help

╔═══════════════════════════════════════════════════════════════════════════╗
║                            PYTHON API USAGE                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

from src.models import ModelManager
from src.analogy_tests import test_analogy, run_analogy_test_suite

# Load model
manager = ModelManager()
model = manager.load_word2vec_google_news()

# Test single analogy
test_analogy(model, "man", "woman", "king", "queen")

# Run full test suite
results = run_analogy_test_suite(model)

╔═══════════════════════════════════════════════════════════════════════════╗
║                              DESIGN BENEFITS                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

✅ Modular & Reusable      → src/ package with clean separation
✅ Single Entry Point       → analogy.py for all CLI operations
✅ Auto-downloads           → Models download automatically via gensim
✅ Extensible              → Easy to add new models/tests
✅ Professional Structure   → Follows Python best practices
✅ Legacy Compatible       → Old files preserved for reference

""")
