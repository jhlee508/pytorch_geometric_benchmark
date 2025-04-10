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
  wikipedia-20070206
  twitter7
"
NODE_NUM="
  Cora:2708
  CiteSeer:3327
  PubMed:19717
  NELL:65755
  Reddit:232965 
  Yelp:716847
  Amazon:1569960
  ogbn-products:2449029
  ogbn-papers100M:111059956
  wikipedia-20070206:3566907
  twitter7:61578415
"
FULL_BS_OF_SMALL_GRAPH_DATASET="
  Cora:2708
  CiteSeer:3327
  PubMed:19717
  NELL:65755
  Reddit:232965 
  Yelp:716847
"
MINI_BS_OF_LARGE_GRAPH_DATASET="
  Amazon:12266
  ogbn-products:122452
  ogbn-papers100M:1
  wikipedia-20070206:1784
  twitter7:1 
"
SMALL_GRAPH_DATASET="
  Cora
  CiteSeer
  PubMed
  NELL
  Reddit 
  Yelp
" 
LARGE_GRAPH_DATASET="
  Amazon
  ogbn-products
  ogbn-papers100M
  wikipedia-20070206
  twitter7
"
TEST_GRAPH_DATASET="
  ogbn-papers100M
"

# Small Graphs
for DATASET in $SMALL_GRAPH_DATASET
do
  echo "======================START======================="
  echo "> Running GCN inference for Graph: $DATASET"

  for DS_INFO in $FULL_BS_OF_SMALL_GRAPH_DATASET
  do
    KEY=$(echo "$DS_INFO" | cut -d':' -f1)
    VAL=$(echo "$DS_INFO" | cut -d':' -f2)

    if [ "$KEY" = "$DATASET" ]; then
      BS=$VAL
      break
    fi
  done

  echo "> Using batch size $BS for $DATASET"

  start_time=$(date +%s.%N)

  python -u $CODE_DIR/inference_benchmark.py \
    --models=gcn \
    --num-layers=2 \
    --num-hidden-channels=256 \
    --datasets=$DATASET \
    --device=cuda \
    --eval-batch-sizes=$BS \
    --measure-load-time \
    --use-sparse-tensor \
    --full-batch \
    
    # --profile \
    # --export-chrome-trace \
    # --write-csv=prof \

  end_time=$(date +%s.%N)

  elapsed=$(echo "$end_time - $start_time" | bc)
  echo ">>> Wall-clock time: $elapsed seconds"

  echo "=======================END========================"
done

# Large Graphs
for DATASET in $LARGE_GRAPH_DATASET
do
  echo "======================START======================="
  echo "> Running GCN inference for Graph: $DATASET"

  for DS_INFO in $MINI_BS_OF_LARGE_GRAPH_DATASET
  do
    KEY=$(echo "$DS_INFO" | cut -d':' -f1)
    VAL=$(echo "$DS_INFO" | cut -d':' -f2)

    if [ "$KEY" = "$DATASET" ]; then
      BS=$VAL
      break
    fi
  done

  echo "> Using batch size $BS for $DATASET"

  start_time=$(date +%s.%N)

  python -u $CODE_DIR/inference_benchmark.py \
    --models=gcn \
    --num-layers=2 \
    --num-hidden-channels=256 \
    --datasets=$DATASET \
    --device=cuda \
    --eval-batch-sizes=$BS \
    --measure-load-time \
    --use-sparse-tensor \
    
    # --full-batch \

    # --profile \
    # --export-chrome-trace \
    # --write-csv=prof \

  end_time=$(date +%s.%N)

  elapsed=$(echo "$end_time - $start_time" | bc)
  echo ">>> Wall-clock time: $elapsed seconds"

  echo "=======================END========================"
done