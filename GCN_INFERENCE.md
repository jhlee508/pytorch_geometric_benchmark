# PyG Installation 

## Setup (PyG)

```bash
# Create conda env
conda create -n pyg python=3.11
conda activate pyg

# PyTorch 2.5.0 on CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyG (torch_geometric) and additional libraries
pip install torch_geometric
pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Run Benchmark (GCN Inference)

```bash
bash gcn_inference.sh
```
