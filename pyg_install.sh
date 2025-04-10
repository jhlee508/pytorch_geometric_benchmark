#!/bin/bash

start_time=$(date +%s)
echo "Starting installation at $(date)"


# PyTorch 2.5.0 on CUDA 12.4
pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyG (torch_geometric) and additional libraries
pip3 install torch_geometric==2.5.0
pip3 install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

pip install ogb

pip3 install xxhash

echo "PyTorch and PyG installed successfully."
echo "Time taken (minutes and seconds): $(( ( $(date +%s) - $start_time ) / 60 ))m $(( ( $(date +%s) - $start_time ) % 60 ))s"