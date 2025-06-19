#!/bin/bash

set -e  # Ferma lo script in caso di errori

# Variabili
PYTHON_VERSION="3.12.0"
ENV_NAME="torch_venv"
#VENV_PATH="$HOME/$ENV_NAME"
VENV_PATH="$ENV_NAME"
KERNEL_NAME="pytorch312"
DISPLAY_NAME="Python 3.12 (PyTorch)"
GPU_TEST_SCRIPT="$HOME/SageMaker/Bees_Knees/src/test/gpu-test.py"

# Controllo argomento
if [[ "$1" == "install" ]]; then
    echo "🐍 Creating virtual environment '$ENV_NAME'..."
    python${PYTHON_VERSION%.*} -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"

    echo "📦 Upgrading pip and installing PyTorch with CUDA..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "🧠 Installing Jupyter kernel..."
    pip install ipykernel
    python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

    echo "✅ Installation complete. To activate the environment manually:"
    echo "    source $VENV_PATH/bin/activate"
else
    # Solo esecuzione del test GPU
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "⚠️  Virtual environment not found. Run '$0 install' first."
        exit 1
    fi
    echo "🔄 Activating virtual environment '$ENV_NAME'..."
    source "$VENV_PATH/bin/activate"
fi

# Esegui il test GPU
echo "🚀 Running GPU test script..."
python "$GPU_TEST_SCRIPT"