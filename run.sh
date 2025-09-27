#!/bin/bash
echo "=================================================="
echo "CS2 Cheat Detection System - Starting Server"
echo "=================================================="
echo

# First check if conda exists in the typical location
if [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Found conda at $HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    echo "Found conda at $HOME/anaconda3"
    export PATH="$HOME/anaconda3/bin:$PATH"
    CONDA_BASE="$HOME/anaconda3"
elif command -v conda &> /dev/null; then
    echo "Conda found in PATH"
    CONDA_BASE=$(conda info --base)
else
    echo "ERROR: Conda not found"
    echo "Please run ./install.sh first to set up the environment"
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Source conda.sh for activation
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh"
    echo "Your conda installation may be incomplete"
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Check if cs2-detect-env exists
if ! conda env list | grep -q "cs2-detect-env"; then
    echo "ERROR: cs2-detect-env environment not found"
    echo "Please run ./install.sh first to set up the environment"
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "Activating cs2-detect-env environment..."

# Activate the environment
conda activate cs2-detect-env

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "Starting the CS2 Cheat Detection web interface..."
echo
echo "Once started, open your browser and go to:"
echo "http://localhost:5000"
echo
echo "Press Ctrl+C to stop the server when done."
echo

# Run the application
python main.py

echo
echo "Server stopped."
echo "Press Enter to exit..."
read