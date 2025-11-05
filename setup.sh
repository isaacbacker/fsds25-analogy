#!/bin/bash

# Setup script for fsds25-analogy project
# This script creates a virtual environment and installs requirements

set -e  # Exit on error

echo "🚀 Starting setup for fsds25-analogy project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at ./$VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "📦 Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at ./$VENV_DIR"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📚 Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ Requirements installed successfully"
else
    echo "⚠️  Warning: requirements.txt not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
