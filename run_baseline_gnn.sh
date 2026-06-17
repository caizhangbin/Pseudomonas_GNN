#!/bin/bash

set -e
conda activate chem_gnn

GRAPHS="AID_1527_fppool_rdkit/graphs.pt"
EPOCHS=100
BATCH_SIZE=32
POOLING="mean"
HIDDEN_DIM=128
NUM_LAYERS=3
LR=0.001

echo "Starting baseline GNN runs..."
echo "Graphs file: $GRAPHS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Pooling: $POOLING"
echo ""

echo "======================================"
echo "Running baseline GCN"
echo "======================================"

python train_baseline_gnn.py \
  --graphs "$GRAPHS" \
  --output baseline_gcn_output \
  --gnn-type GCN \
  --pooling "$POOLING" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  2>&1 | tee baseline_gcn.log


echo ""
echo "======================================"
echo "Running baseline GIN"
echo "======================================"

python train_baseline_gnn.py \
  --graphs "$GRAPHS" \
  --output baseline_gin_output \
  --gnn-type GIN \
  --pooling "$POOLING" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  2>&1 | tee baseline_gin.log


echo ""
echo "======================================"
echo "Running baseline GINE"
echo "======================================"

python train_baseline_gnn.py \
  --graphs "$GRAPHS" \
  --output baseline_gine_output \
  --gnn-type GINE \
  --pooling "$POOLING" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --lr "$LR" \
  2>&1 | tee baseline_gine.log


echo ""
echo "All baseline GNN runs finished."
echo "Outputs:"
echo "  baseline_gcn_output/"
echo "  baseline_gin_output/"
echo "  baseline_gine_output/"
echo ""
echo "Logs:"
echo "  baseline_gcn.log"
echo "  baseline_gin.log"
echo "  baseline_gine.log"