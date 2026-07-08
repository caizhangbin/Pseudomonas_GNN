#!/usr/bin/env python3

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


# ------------------------------------------------------------
# Import model class from your training script
# ------------------------------------------------------------

def import_model_class(train_script_path):
    train_script_path = Path(train_script_path)

    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    spec = importlib.util.spec_from_file_location(
        "train_baseline_gnns_module",
        str(train_script_path)
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "GNNWithOptionalFP"):
        raise AttributeError(
            "Could not find GNNWithOptionalFP in the training script."
        )

    return module.GNNWithOptionalFP


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def run_command(cmd, step_name):
    print(f"\n=== {step_name} ===")
    print(" ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed.")


def read_smiles_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")

    smiles = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line:
                smiles.append(line)

    return smiles


def print_skipped_smiles(skipped_path):
    """
    Safely print skipped SMILES if the skipped_fppool_errors.csv file
    exists and contains readable rows.

    This avoids pandas.errors.EmptyDataError when the file exists but is empty.
    """

    skipped_path = Path(skipped_path)

    if not skipped_path.exists():
        return

    if skipped_path.stat().st_size == 0:
        return

    try:
        skipped_df = pd.read_csv(skipped_path)
    except pd.errors.EmptyDataError:
        return

    if len(skipped_df) > 0:
        print("\nSome SMILES were skipped during graph creation:")
        print(skipped_df)


def create_prediction_graphs(
    smiles_list,
    prepare_script,
    fixed_morgan_script,
    fppool_path,
    radius,
    n_bits,
    temp_dir,
):
    """
    Uses your existing workflow:

    1. prepare_fppool_graphs.py
    2. make_fixed_morgan_graphs.py

    A dummy label column is added because prepare_fppool_graphs.py expects labels.
    The label is not used for prediction.
    """

    temp_dir = Path(temp_dir)

    input_csv = temp_dir / "input_smiles.csv"
    fppool_dir = temp_dir / "fppool_graphs"
    fixed_dir = temp_dir / "fixed_morgan_graphs"

    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "label": [0] * len(smiles_list),
        }
    )

    df.to_csv(input_csv, index=False)

    cmd_prepare = [
        sys.executable,
        str(prepare_script),
        "-i",
        str(input_csv),
        "-o",
        str(fppool_dir),
        "--fppool-path",
        str(fppool_path),
        "--fp-types",
        "MorganFP",
        "--smiles-col",
        "smiles",
        "--label-col",
        "label",
    ]

    run_command(cmd_prepare, "Preparing FPPool Morgan graphs")

    cmd_fixed = [
        sys.executable,
        str(fixed_morgan_script),
        "--input-graphs",
        str(fppool_dir / "graphs.pt"),
        "--output-dir",
        str(fixed_dir),
        "--radius",
        str(radius),
        "--n-bits",
        str(n_bits),
    ]

    run_command(cmd_fixed, "Creating fixed MorganFP graphs")

    graphs_path = fixed_dir / "graphs.pt"

    if not graphs_path.exists():
        raise FileNotFoundError(
            f"Expected graph file was not created: {graphs_path}"
        )

    skipped_path = fppool_dir / "skipped_fppool_errors.csv"

    return graphs_path, skipped_path


# ------------------------------------------------------------
# Model loading and prediction
# ------------------------------------------------------------

def load_model(model_path, train_script_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    GNNWithOptionalFP = import_model_class(train_script_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    args = checkpoint.get("args", {})

    model = GNNWithOptionalFP(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        fp_dim=checkpoint["fp_dim"],
        hidden_dim=int(args.get("hidden_dim", 64)),
        num_layers=int(args.get("num_layers", 2)),
        dropout=float(args.get("dropout", 0.5)),
        gnn_type=args.get("gnn_type", "GIN"),
        pooling=args.get("pooling", "mean"),
        num_classes=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    model_info = {
        "model_path": str(model_path),
        "device": str(device),
        "gnn_type": args.get("gnn_type", "unknown"),
        "pooling": args.get("pooling", "unknown"),
        "hidden_dim": args.get("hidden_dim", "unknown"),
        "num_layers": args.get("num_layers", "unknown"),
        "dropout": args.get("dropout", "unknown"),
        "fp_dim": checkpoint.get("fp_dim", "unknown"),
        "checkpoint_selected_threshold": checkpoint.get("selected_threshold", None),
    }

    return model, device, model_info


def predict_from_graphs(model, device, graphs_path, threshold=0.8, batch_size=64):
    graphs = torch.load(
        graphs_path,
        map_location=device,
        weights_only=False
    )

    if not isinstance(graphs, list):
        raise ValueError(
            "graphs.pt must contain a list of PyTorch Geometric Data objects."
        )

    if len(graphs) == 0:
        raise ValueError(
            "No valid graphs were created. Check skipped_fppool_errors.csv."
        )

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits = model(batch)
            prob_active = F.softmax(logits, dim=1)[:, 1]

            probabilities.extend(prob_active.cpu().tolist())

    rows = []

    for i, (graph, prob_active) in enumerate(zip(graphs, probabilities)):
        input_smiles = getattr(graph, "original_smiles", None)
        canonical_smiles = getattr(graph, "smiles", None)

        if input_smiles is None:
            input_smiles = canonical_smiles

        prediction = "Active" if prob_active >= threshold else "Inactive"

        if prediction == "Active":
            confidence = prob_active
        else:
            confidence = 1.0 - prob_active

        rows.append(
            {
                "index": i,
                "input_smiles": input_smiles,
                "canonical_smiles": canonical_smiles,
                "probability_active": prob_active,
                "threshold": threshold,
                "prediction": prediction,
                "confidence": confidence,
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "probability_active",
        ascending=False
    ).reset_index(drop=True)

    return results


# ------------------------------------------------------------
# Display and save functions
# ------------------------------------------------------------

def print_results(results):
    print("\n=== Prediction Results ===")

    for _, row in results.iterrows():
        print("\nSMILES:", row["input_smiles"])
        print("Canonical SMILES:", row["canonical_smiles"])
        print("Prediction:", row["prediction"])
        print(f"Probability active: {row['probability_active']:.4f}")
        print(f"Threshold used: {row['threshold']:.2f}")
        print(f"Confidence: {row['confidence']:.4f}")


def save_outputs(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "smiles_predictions.csv"
    json_path = output_dir / "smiles_predictions.json"

    results.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(
            results.to_dict(orient="records"),
            f,
            indent=2
        )

    print(f"\nSaved CSV predictions to: {csv_path}")
    print(f"Saved JSON predictions to: {json_path}")


# ------------------------------------------------------------
# Interactive mode
# ------------------------------------------------------------

def interactive_smiles_input():
    print("\nEnter SMILES strings one per line.")
    print("Press Enter on an empty line when finished.\n")

    smiles = []

    while True:
        line = input("SMILES > ").strip()

        if line == "":
            break

        smiles.append(line)

    return smiles


def run_prediction_pipeline(args, smiles_list):
    if len(smiles_list) == 0:
        print("No SMILES were provided.")
        return

    temp_parent = Path(args.temp_dir)
    temp_parent.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.mkdtemp(
        prefix="morgan_gnn_prediction_",
        dir=temp_parent
    )

    try:
        graphs_path, skipped_path = create_prediction_graphs(
            smiles_list=smiles_list,
            prepare_script=args.prepare_script,
            fixed_morgan_script=args.fixed_morgan_script,
            fppool_path=args.fppool_path,
            radius=args.radius,
            n_bits=args.n_bits,
            temp_dir=temp_dir,
        )

        model, device, model_info = load_model(
            model_path=args.model_path,
            train_script_path=args.train_script,
        )

        print("\n=== Loaded Model ===")
        print(json.dumps(model_info, indent=2))

        results = predict_from_graphs(
            model=model,
            device=device,
            graphs_path=graphs_path,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )

        print_results(results)

        print_skipped_smiles(skipped_path)

        if args.save_predictions:
            save_outputs(results, args.output_dir)

    finally:
        if args.keep_temp:
            print(f"\nTemporary files kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_interactive_menu(args):
    while True:
        print("\n======================================")
        print("MorganFP + GNN Activity Predictor")
        print("======================================")
        print("1. Predict activity from custom SMILES")
        print("2. Run example predictions")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            smiles_list = interactive_smiles_input()
            run_prediction_pipeline(args, smiles_list)

        elif choice == "2":
            example_smiles = [
                "CCO",
                "c1ccccc1",
                "CC(=O)O",
            ]

            print("\nRunning examples:")

            for s in example_smiles:
                print(s)

            run_prediction_pipeline(args, example_smiles)

        elif choice == "3":
            print("Exiting.")
            break

        else:
            print("Invalid option. Please choose 1, 2, or 3.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repo-style SMILES prediction script for MorganFP + GNN model."
    )

    parser.add_argument(
        "--model-path",
        default="baseline_gin_morgan_regularized_v1/final_model.pt",
        help="Path to trained model checkpoint."
    )

    parser.add_argument(
        "--train-script",
        default="train_baseline_gnns.py",
        help="Path to train_baseline_gnns.py containing GNNWithOptionalFP."
    )

    parser.add_argument(
        "--prepare-script",
        default="prepare_fppool_graphs.py",
        help="Path to prepare_fppool_graphs.py."
    )

    parser.add_argument(
        "--fixed-morgan-script",
        default="make_fixed_morgan_graphs.py",
        help="Path to make_fixed_morgan_graphs.py."
    )

    parser.add_argument(
        "--fppool-path",
        default="./FPPool",
        help="Path to the cloned FPPool repository."
    )

    parser.add_argument(
        "--smiles",
        nargs="*",
        default=None,
        help="One or more SMILES strings."
    )

    parser.add_argument(
        "--smiles-file",
        default=None,
        help="Text file containing one SMILES string per line."
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run terminal menu mode."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Classification threshold. Default is 0.8."
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius."
    )

    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit length."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prediction batch size."
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV and JSON."
    )

    parser.add_argument(
        "--output-dir",
        default="prediction_results",
        help="Output directory for saved predictions."
    )

    parser.add_argument(
        "--temp-dir",
        default="prediction_temp",
        help="Directory for temporary graph files."
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary graph files for debugging."
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive_menu(args)
        return

    smiles_list = []

    if args.smiles:
        smiles_list.extend(args.smiles)

    if args.smiles_file:
        smiles_list.extend(read_smiles_file(args.smiles_file))

    if len(smiles_list) == 0:
        print("No SMILES provided. Starting interactive mode instead.")
        run_interactive_menu(args)
        return

    run_prediction_pipeline(args, smiles_list)


if __name__ == "__main__":
    main()