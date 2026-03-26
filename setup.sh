#!/bin/bash
# MedClear — Setup Script
# Run: chmod +x setup.sh && ./setup.sh

echo "========================================="
echo "  MedClear Setup"
echo "========================================="

# Check for NVIDIA GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Ensure CUDA drivers are installed."
fi

# Check Python
echo ""
echo "Python version:"
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0
pip install datasets>=2.18.0
pip install peft>=0.10.0
pip install bitsandbytes>=0.43.0
pip install accelerate>=0.29.0
pip install trl>=0.8.0
pip install rouge-score
pip install nltk
pip install textstat
pip install sentencepiece
pip install protobuf

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Verify installation
echo ""
echo "========================================="
echo "  Verification"
echo "========================================="
python3 -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers
print(f'Transformers:   {transformers.__version__}')

import peft
print(f'PEFT:           {peft.__version__}')

import bitsandbytes
print(f'BitsAndBytes:   {bitsandbytes.__version__}')

import trl
print(f'TRL:            {trl.__version__}')

print()
print('All dependencies installed successfully!')
"

echo ""
echo "========================================="
echo "  Ready! Run: python cochrane_plainlang_finetune.py --mode all"
echo "========================================="
