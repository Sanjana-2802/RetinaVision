#!/usr/bin/env bash
# Cloud build script for fast, lightweight deployment
set -e

echo "==> Step 1: Upgrading pip..."
pip install --upgrade pip

echo "==> Step 2: Installing lightweight PyTorch CPU-only (skips 4.5GB NVIDIA packages)..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Step 3: Installing application dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "==> Step 4: Pre-warming presets and clinical storage directories..."
python -c "from app.ml.presets import generate_preset_images; generate_preset_images()"

echo "==> Build successful! Ready for cloud execution."
