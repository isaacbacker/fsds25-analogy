from setuptools import setup, find_packages

setup(
    name="fsds25-analogy",
    version="0.1.0",
    description="A project for exploring word analogies using embeddings and LLMs",
    author="Oxford Internet Institute",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "gensim>=4.3.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
)
