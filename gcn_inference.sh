#!/bin/bash

export PYTHONPATH=$HOME/fpga-gpu/pytorch_geometric_benchmark

CODE_DIR=$HOME/fpga-gpu/pytorch_geometric_benchmark/benchmark/inference

ALL_GRAPH_DATASET="
  Cora
  CiteSeer
  PubMed
  NELL
  Reddit 
  Yelp
  Amazon
  ogbn-products
  ogbn-papers100M
"
FULL_BS_OF_GRAPH_DATASET="
  Cora:2708
  CiteSeer:3327
  PubMed:19717
  NELL:65755
  Reddit:232965 
  Yelp:716847
  Amazon:1569960
  ogbn-products:2449029
  ogbn-papers100M:111059956
"
MINI_BS_OF_GRAPH_DATASET="
  Cora:2708
  CiteSeer:3327
  PubMed:19717
  NELL:65755
  Reddit:232965 
  Yelp:716847
  Amazon:392490
  ogbn-products:2449029
  ogbn-papers100M:111059956
"
GRAPH_DATASET="
  Cora
  CiteSeer
  PubMed
  NELL
  Reddit
  Yelp
" 

for DATASET in $GRAPH_DATASET
do
  echo "======================START======================="
  echo "> Running GCN inference for Graph: $DATASET"

  for DS_INFO in $MINI_BS_OF_GRAPH_DATASET
  do
    KEY=$(echo "$DS_INFO" | cut -d':' -f1)
    VAL=$(echo "$DS_INFO" | cut -d':' -f2)

    if [ "$KEY" = "$DATASET" ]; then
      BS=$VAL
      break
    fi
  done

  echo "> Using batch size $BS for $DATASET"

  srun -p PV-Short --exclusive --gres=gpu:1 \
    python -u $CODE_DIR/inference_benchmark.py \
      --models=gcn \
      --num-layers=2 \
      --num-hidden-channels=256 \
      --datasets=$DATASET \
      --device=cuda \
      --warmup=0 \
      --eval-batch-sizes=$BS \
      --measure-load-time \
      --full-batch \
      --use-sparse-tensor \
      
      # --profile \
      # --export-chrome-trace \
      # --write-csv=prof \

  echo "=======================END========================"
done