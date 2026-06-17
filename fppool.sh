#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chem_gnn

python prepare_fppool_graphs.py \
  -i AID_1527_clean/cleaned_data.csv \
  -o AID_1527_fppool_rdkit \
  --fppool-path ./FPPooling \
  --fp-types RdkitFP