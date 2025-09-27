#!/bin/bash
echo "=================================================="
echo "CS2 Cheat Detection System - Linux/Mac Setup"
echo "=================================================="
echo

# First check if conda exists in the typical location
if [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Found conda at $HOME/miniconda3"
    # Add to PATH for this session
    export PATH="$HOME/miniconda3/bin:$PATH"
    # Source conda for this session
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    echo "Found conda at $HOME/anaconda3"
    # Add to PATH for this session
    export PATH="$HOME/anaconda3/bin:$PATH"
    # Source conda for this session
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    echo "Conda found in PATH"
else
    echo "Conda is not installed. Installing Miniconda..."
    echo

    # Detect OS and architecture
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Check if Apple Silicon
        if [[ $(uname -m) == 'arm64' ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        echo "ERROR: Unsupported OS type: $OSTYPE"
        echo "Please install Miniconda manually from https://docs.conda.io/en/latest/miniconda.html"
        echo "Press Enter to exit..."
        read
        exit 1
    fi

    # Download and install Miniconda
    echo "Downloading Miniconda..."
    curl -o miniconda.sh $MINICONDA_URL

    echo "Installing Miniconda..."
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh

    # Add conda to PATH for this session
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"

    # Initialize conda for bash
    conda init bash

    echo
    echo "Miniconda installed successfully!"
    echo "Continuing with environment setup..."
    echo
fi

echo "Conda is available! Setting up environment..."
echo

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi | grep "NVIDIA GeForce\|NVIDIA RTX\|NVIDIA Quadro" | head -1
    echo "CUDA-enabled PyTorch will be installed for GPU acceleration."
else
    echo "WARNING: No NVIDIA GPU detected. Installing CPU-only PyTorch."
    echo "For GPU acceleration, please install NVIDIA drivers first."
fi
echo

# Check if cs2-detect-env already exists
if conda env list | grep -q "cs2-detect-env"; then
    echo "Environment 'cs2-detect-env' already exists."
    echo "Do you want to:"
    echo "1) Use existing environment (recommended if it's working)"
    echo "2) Update the existing environment"
    echo "3) Remove and recreate the environment"
    echo "4) Cancel installation"
    echo
    read -p "Enter choice (1/2/3/4): " choice

    case $choice in
        1)
            echo "Using existing environment..."
            ;;
        2)
            echo "Updating existing environment..."
            conda env update -f environment.yml
            ;;
        3)
            echo "Removing existing environment..."
            conda env remove -n cs2-detect-env -y
            echo "Creating new environment..."
            conda env create -f environment.yml
            ;;
        4)
            echo "Installation cancelled."
            echo "Press Enter to exit..."
            read
            exit 0
            ;;
        *)
            echo "Invalid choice. Installation cancelled."
            echo "Press Enter to exit..."
            read
            exit 1
            ;;
    esac
else
    echo "Creating conda environment 'cs2-detect-env'..."
    echo "This may take 10-15 minutes depending on your internet connection..."
    echo
    conda env create -f environment.yml
fi

if [ $? -ne 0 ]; then
    # Only show error if we actually tried to create/update
    if [[ "$choice" != "1" ]]; then
        echo
        echo "ERROR: Environment setup failed"
        echo "Please check the error messages above"
        echo "Press Enter to exit..."
        read
        exit 1
    fi
fi

# Update main.py to use the correct Python path
echo
echo "Updating configuration..."

# Get the conda environment Python path
CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3}"
ENV_PYTHON="$CONDA_PREFIX/envs/cs2-detect-env/bin/python"

# Check if we need to update main.py (handle any hardcoded Python paths)
if [ -f main.py ]; then
    # Look for any hardcoded Python paths in main.py and replace them
    if grep -q "/home/.*/miniconda3/envs/cs2-detect-env/bin/python\|/home/.*/anaconda3/envs/cs2-detect-env/bin/python" main.py; then
        # Replace any hardcoded conda environment path with the actual one
        sed -i "s|/home/.*/miniconda3/envs/cs2-detect-env/bin/python|$ENV_PYTHON|g" main.py
        sed -i "s|/home/.*/anaconda3/envs/cs2-detect-env/bin/python|$ENV_PYTHON|g" main.py
        echo "Updated hardcoded Python paths in main.py to: $ENV_PYTHON"
    else
        echo "No hardcoded Python paths found in main.py (using dynamic detection)"
    fi
fi

echo
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo
echo "To start the CS2 Cheat Detection System:"
echo "1. Run: ./run.sh"
echo "   (The run script will automatically activate the environment)"
echo "2. Open your browser to http://localhost:5000"
echo
echo "Note: The model weights (.pth files) need to be placed in:"
echo "  deepcheat/VideoMAEv2/output/"
echo
echo "Press Enter to exit..."
read

# Make run script executable
chmod +x run.sh