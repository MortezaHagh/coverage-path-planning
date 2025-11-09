#!/bin/bash
# Quick installation script for Coverage Path Planning package

set -e  # Exit on error

echo "=================================="
echo "Coverage Path Planning Installer"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.8+"; exit 1; }

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode
echo ""
echo "Installing package in editable mode..."
pip install -e .

echo ""
echo "=================================="
echo "Installation Complete! ✓"
echo "=================================="
echo ""
echo "To use the package:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the pipeline: python main.py Map2"
echo "  3. Or use command: covpp Map2"
echo ""
echo "To deactivate virtual environment: deactivate"
echo ""
