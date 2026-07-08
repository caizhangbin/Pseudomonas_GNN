# Pseudomonas GNN

Graph neural network workflow for predicting antibacterial activity against *Pseudomonas aeruginosa* from molecular SMILES strings.

This repository contains preprocessing, graph construction, model training, and prediction scripts for molecular activity classification using PyTorch Geometric, RDKit, and FPPool-based molecular fingerprint pooling.

## Project Overview

This project uses molecular structures represented as SMILES strings to train graph neural network models for binary antibacterial activity prediction.

The workflow includes:

1. Cleaning raw PubChem assay data
2. Converting SMILES into canonical SMILES and SMARTS
3. Creating molecular graph objects
4. Adding molecular fingerprint information using FPPool
5. Training baseline GNN models
6. Training GNN + FPPool models
7. Predicting activity for new SMILES strings

The main task is binary classification:

- `Active`: molecule shows antibacterial activity
- `Inactive`: molecule does not show antibacterial activity

## Repository Structure

```text
Pseudomonas_GNN/
├── environment.yaml
├── prepare_data.py
├── prepare_fppool_graphs.py
├── prepare_fppool_smarts_graphs.py
├── train_baseline_gnns.py
├── train_fppool_gnn.py
├── make_fixed_morgan_graphs.py
├── predict_morgan_gnn.py
├── check_overfitting.py
├── run_baseline_gnn.sh
├── fppool.sh
└── LICENSE
```

## Main Scripts

### `prepare_data.py`

Cleans raw molecular activity data.

Main functions:

- Reads a raw CSV file
- Finds SMILES and activity columns
- Converts activity outcomes into binary labels
- Removes unusable labels
- Validates SMILES with RDKit
- Generates canonical SMILES
- Generates SMARTS
- Removes duplicate molecules with conflicting labels
- Saves cleaned datasets and a cleaning summary

Example:

```bash
python prepare_data.py \
  -i AID_1527_datatable.csv \
  -o AID_1527_clean \
  --skiprows 1,2,3,4
```

Expected main output:

```text
AID_1527_clean/cleaned_data.csv
AID_1527_clean/cleaning_summary.txt
AID_1527_clean/removed_invalid_smiles.csv
AID_1527_clean/removed_duplicate_label_conflicts.csv
```

### `prepare_fppool_graphs.py`

Creates PyTorch Geometric molecular graph objects using FPPool-compatible fingerprint features.

Main functions:

- Loads cleaned molecular data
- Validates canonical SMILES
- Builds PyTorch Geometric `Data` objects
- Adds node features, edge features, labels, and fingerprint features
- Saves processed graphs as `graphs.pt`

Example:

```bash
python prepare_fppool_graphs.py \
  -i AID_1527_clean/cleaned_data.csv \
  -o AID_1527_fppool_rdkit \
  --fppool-path ./FPPooling \
  --fp-types RdkitFP
```

Expected output:

```text
AID_1527_fppool_rdkit/graphs.pt
AID_1527_fppool_rdkit/graph_summary.json
AID_1527_fppool_rdkit/skipped_fppool_errors.csv
```

Supported fingerprint types include:

```text
MorganFP
RdkitFP
EstateFP
MACCSFP
PubChemFP
RGroupFP
FragmentFP
```

### `train_baseline_gnns.py`

Trains baseline graph neural network models using molecular graphs and optional stored fingerprint features.

Supported GNN types:

- `GCN`
- `GIN`
- `GINE`

Example:

```bash
python train_baseline_gnns.py \
  --graphs AID_1527_fppool_rdkit/graphs.pt \
  --output baseline_gcn_output \
  --gnn-type GCN \
  --pooling mean \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 3 \
  --lr 0.001
```

The script performs:

- Stratified train/validation/test split
- Model training
- Validation-based model selection
- Test set evaluation
- Metric export

Typical outputs:

```text
baseline_gcn_output/
├── best.pt
├── final.pt
├── history.json
└── summary.json
```

### `train_fppool_gnn.py`

Trains a GNN model combined with FPPool attention-based molecular fingerprint pooling.

Example:

```bash
python train_fppool_gnn.py \
  --graphs AID_1527_fppool_rdkit/graphs.pt \
  --fppool-path ./FPPooling \
  --output fppool_gcn_output \
  --gnn-type GCN \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --layers 3 \
  --lr 0.001
```

The model includes:

- Molecular graph encoder
- Atom-level GNN representation
- FPPool fingerprint-guided pooling
- Binary activity classifier

Typical outputs:

```text
fppool_gcn_output/
├── best.pt
├── final.pt
├── history.json
└── summary.json
```

### `make_fixed_morgan_graphs.py`

Converts existing graph objects into graphs with fixed-size Morgan fingerprints stored in `data.fp`.

Example:

```bash
python make_fixed_morgan_graphs.py \
  --input-graphs AID_1527_fppool_rdkit/graphs.pt \
  --output-dir AID_1527_morgan_fixed \
  --radius 2 \
  --n-bits 2048
```

Expected output:

```text
AID_1527_morgan_fixed/graphs.pt
AID_1527_morgan_fixed/fixed_morgan_summary.json
```

### `predict_morgan_gnn.py`

Predicts antibacterial activity for new molecules from SMILES strings using a trained MorganFP + GNN model.

Example using SMILES directly:

```bash
python predict_morgan_gnn.py \
  --model-path baseline_gin_morgan_regularized_v1/final_model.pt \
  --smiles "CCO" "c1ccccc1" "CC(=O)O" \
  --threshold 0.8 \
  --save-predictions
```

Example using a SMILES text file:

```bash
python predict_morgan_gnn.py \
  --model-path baseline_gin_morgan_regularized_v1/final_model.pt \
  --smiles-file input_smiles.txt \
  --threshold 0.8 \
  --save-predictions
```

Example interactive mode:

```bash
python predict_morgan_gnn.py --interactive
```

Expected output:

```text
prediction_results/smiles_predictions.csv
prediction_results/smiles_predictions.json
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate chem_gnn
```

The environment includes:

- Python 3.10
- pandas
- numpy
- scikit-learn
- RDKit
- PyTorch
- PyTorch Geometric
- CUDA-enabled PyTorch dependencies, if available

## FPPool Setup

Some scripts require the external FPPool repository.

Clone or place the FPPool/FPPooling repository inside this project folder:

```bash
git clone https://github.com/shenwxlab/FPPool.git FPPooling
```

Then use:

```bash
--fppool-path ./FPPooling
```

or adjust the path depending on your local folder name.

## Full Example Workflow

### Step 1: Clean raw assay data

```bash
python prepare_data.py \
  -i AID_1527_datatable.csv \
  -o AID_1527_clean \
  --skiprows 1,2,3,4
```

### Step 2: Create FPPool molecular graphs

```bash
python prepare_fppool_graphs.py \
  -i AID_1527_clean/cleaned_data.csv \
  -o AID_1527_fppool_rdkit \
  --fppool-path ./FPPooling \
  --fp-types RdkitFP
```

### Step 3: Train a baseline GNN

```bash
python train_baseline_gnns.py \
  --graphs AID_1527_fppool_rdkit/graphs.pt \
  --output baseline_gcn_output \
  --gnn-type GCN \
  --pooling mean \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 3 \
  --lr 0.001
```

### Step 4: Train a GNN + FPPool model

```bash
python train_fppool_gnn.py \
  --graphs AID_1527_fppool_rdkit/graphs.pt \
  --fppool-path ./FPPooling \
  --output fppool_gcn_output \
  --gnn-type GCN \
  --epochs 100 \
  --batch-size 32
```

### Step 5: Predict activity for new SMILES

```bash
python predict_morgan_gnn.py \
  --model-path baseline_gin_morgan_regularized_v1/final_model.pt \
  --smiles "CCO" "c1ccccc1" \
  --threshold 0.8 \
  --save-predictions
```

## Output Metrics

Training scripts report common binary classification metrics, including:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Specificity
- F1 score
- ROC-AUC
- Average precision
- Confusion matrix counts

Because the dataset is highly imbalanced, balanced accuracy, recall, ROC-AUC, and average precision are especially useful for evaluating model performance.

## Notes

- The dataset is expected to contain molecular SMILES strings and activity labels.
- Activity labels are converted into binary values:
  - `Active` -> `1`
  - `Inactive` -> `0`
- RDKit is used to validate and canonicalize SMILES.
- PyTorch Geometric is used for molecular graph representation.
- FPPool is used for fingerprint-guided graph pooling.
- GINE requires edge attributes, while GCN and GIN can run without using edge attributes directly.



## License

This project is licensed under the MIT License.

