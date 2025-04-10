#!/bin/bash

start_time=$(date +%s)
echo "Starting installation at $(date)"


# PyTorch 2.0.0 on CUDA 11.7
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# Install PyG (torch_geometric) and additional libraries
pip install torch_geometric
pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Downgrade Numpy
pip install 'numpy<2'

# Install OGB
pip install ogb

pip3 install xxhash

echo "PyTorch and PyG installed successfully."
echo "Time taken (minutes and seconds): $(( ( $(date +%s) - $start_time ) / 60 ))m $(( ( $(date +%s) - $start_time ) % 60 ))s"