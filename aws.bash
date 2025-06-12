#!/bin/bash

set -e  # Ferma lo script in caso di errori

# Variabili
PYTHON_VERSION="3.12.0"
ENV_NAME="torch_venv"

echo "🐍 Creating virtual environment '$ENV_NAME'..."
python3.12 -m venv ~/$ENV_NAME
source ~/$ENV_NAME/bin/activate

echo "📦 Upgrading pip and installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "🧠 Installing Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name pytorch312 --display-name "Python 3.12 (PyTorch)"

echo "✅ Installation complete. To activate the environment manually:"
echo "source ~/$ENV_NAME/bin/activate"

echo "🚀 Running GPU test script..."
python ~/Bees_Knees/src/test/gpu-test.py