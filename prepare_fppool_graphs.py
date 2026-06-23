#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import torch

try:
    from rdkit import Chem
    from rdkit import RDLogger
except ImportError:
    print("ERROR: RDKit is not installed.")
    print("Install it with: conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: torch_geometric is not installed.")
    print("Install it with: pip install torch-geometric")
    sys.exit(1)


RDLogger.DisableLog("rdApp.warning")


SUPPORTED_FP_TYPES = {
    "MorganFP",
    "RdkitFP",
    "EstateFP",
    "MACCSFP",
    "PubChemFP",
    "RGroupFP",
    "FragmentFP",
}


def parse_fp_types(text: str) -> List[str]:
    fp_types = [x.strip() for x in text.split(",") if x.strip()]

    if len(fp_types) == 0:
        raise ValueError("No fingerprint types provided.")

    bad = [x for x in fp_types if x not in SUPPORTED_FP_TYPES]
    if bad:
        raise ValueError(
            f"Unsupported fp type(s): {bad}\n"
            f"Supported fp types: {sorted(SUPPORTED_FP_TYPES)}"
        )

    return fp_types


def add_fppool_to_path(fppool_path: str) -> None:
    if not os.path.exists(fppool_path):
        raise FileNotFoundError(
            f"FPPool/FPPooling path not found: {fppool_path}\n"
            f"Example: --fppool-path ./FPPooling"
        )

    abs_path = os.path.abspath(fppool_path)

    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


def import_fppool_featurizer(fppool_path: str):
    add_fppool_to_path(fppool_path)

    try:
        from fppcode.feature.featurizer import GenNodeEdgeFeatures39_WithFP
    except Exception as error:
        raise ImportError(
            "Could not import FPPool featurizer.\n\n"
            "Expected import:\n"
            "    from fppcode.feature.featurizer import GenNodeEdgeFeatures39_WithFP\n\n"
            "Check that your repo path contains:\n"
            "    FPPooling/fppcode/feature/featurizer.py\n\n"
            f"Current --fppool-path: {fppool_path}\n"
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    return GenNodeEdgeFeatures39_WithFP


def find_column(df: pd.DataFrame, candidates: List[str], user_col: Optional[str]) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(
                f"Column '{user_col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return user_col

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not find column automatically.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def validate_molecule(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    if pd.isna(smiles):
        return None, "missing_smiles"

    smiles = str(smiles).strip()

    if smiles == "":
        return None, "empty_smiles"

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, "invalid_smiles"

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    if not canonical:
        return None, "canonicalization_failed"

    return canonical, None


def make_data_object(
    smiles: str,
    label: int,
    row_index: int,
    original_smiles: Optional[str] = None,
    smarts: Optional[str] = None,
) -> Data:
    data = Data()

    # FPPool featurizer expects data.smiles
    data.smiles = smiles

    # Classification label
    data.y = torch.tensor([int(label)], dtype=torch.long)

    # Useful metadata
    data.row_index = torch.tensor([int(row_index)], dtype=torch.long)

    if original_smiles is not None:
        data.original_smiles = str(original_smiles)

    if smarts is not None:
        data.smarts = str(smarts)

    return data


def summarize_graphs(graphs: List[Data]) -> dict:
    if not graphs:
        return {}

    labels = torch.cat([g.y for g in graphs]).view(-1)
    active = int((labels == 1).sum().item())
    inactive = int((labels == 0).sum().item())

    first = graphs[0]

    summary = {
        "num_graphs": len(graphs),
        "active": active,
        "inactive": inactive,
        "node_feature_dim": int(first.x.shape[1]) if hasattr(first, "x") else None,
        "edge_feature_dim": int(first.edge_attr.shape[1]) if hasattr(first, "edge_attr") and first.edge_attr.numel() > 0 else None,
        "fp_shape_first_graph": list(first.fp.shape) if hasattr(first, "fp") else None,
        "fp_length_first_graph": first.fp_length.tolist() if hasattr(first, "fp_length") else None,
        "example_smiles": first.smiles if hasattr(first, "smiles") else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Create PyTorch Geometric graphs using official FPPool fingerprint preaggregation."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input cleaned CSV, e.g. clean_data/cleaned_data.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="fppool_processed",
        help="Output folder.",
    )

    parser.add_argument(
        "--fppool-path",
        default="./FPPooling",
        help="Path to cloned FPPooling/FPPool repo. Default: ./FPPooling",
    )

    parser.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column. Default tries canonical_smiles, SMILES, smiles.",
    )

    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column. Default tries label.",
    )

    parser.add_argument(
        "--smarts-col",
        default=None,
        help="SMARTS column. Default tries smarts if available.",
    )

    parser.add_argument(
        "--fp-types",
        default="RdkitFP",
        help=(
            "Comma-separated FPPool fingerprint types. "
            "Examples: RdkitFP or MorganFP,RdkitFP,MACCSFP"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing, e.g. --limit 1000",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50000,
        help="Save temporary checkpoint every N graphs. Default: 50000",
    )

    parser.add_argument(
        "--no-suppress-fppool-print",
        action="store_true",
        help="Do not suppress print output from the FPPool featurizer.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fp_types = parse_fp_types(args.fp_types)

    print(f"Input CSV: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"FPPool path: {args.fppool_path}")
    print(f"Fingerprint types: {fp_types}")

    print("\nImporting FPPool featurizer...")
    GenNodeEdgeFeatures39_WithFP = import_fppool_featurizer(args.fppool_path)

    featurizer = GenNodeEdgeFeatures39_WithFP(fp_types=fp_types)

    print("\nLoading cleaned CSV...")
    df = pd.read_csv(args.input, low_memory=False)

    smiles_col = find_column(
        df,
        candidates=[
            "canonical_smiles",
            "SMILES",
            "smiles",
            "Canonical SMILES",
            "PUBCHEM_OPENEYE_CAN_SMILES",
            "PUBCHEM_OPENEYE_ISO_SMILES",
        ],
        user_col=args.smiles_col,
    )

    label_col = find_column(
        df,
        candidates=["label", "Label", "activity", "Activity"],
        user_col=args.label_col,
    )

    if args.smarts_col:
        smarts_col = find_column(df, candidates=["smarts"], user_col=args.smarts_col)
    else:
        smarts_col = "smarts" if "smarts" in df.columns else None

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Using first {len(df)} rows because --limit was set.")

    print(f"Using SMILES column: {smiles_col}")
    print(f"Using label column: {label_col}")
    print(f"Using SMARTS column: {smarts_col}")

    graphs = []
    error_rows = []

    print("\nCreating FPPool graphs...")

    for i, row in df.iterrows():
        if len(graphs) > 0 and len(graphs) % 1000 == 0:
            print(f"Created {len(graphs)} graphs...")

        raw_smiles = row[smiles_col]
        label = row[label_col]

        try:
            if pd.isna(label):
                raise ValueError("missing_label")

            label = int(label)

            if label not in {0, 1}:
                raise ValueError(f"label_not_binary: {label}")

            canonical_smiles, mol_error = validate_molecule(raw_smiles)

            if mol_error is not None:
                raise ValueError(mol_error)

            smarts = row[smarts_col] if smarts_col is not None else None

            data = make_data_object(
                smiles=canonical_smiles,
                label=label,
                row_index=int(i),
                original_smiles=raw_smiles,
                smarts=smarts,
            )

            # FPPool's featurizer prints each molecule in the repo version.
            # Suppress that by default because this dataset is large.
            if args.no_suppress_fppool_print:
                data = featurizer(data)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    data = featurizer(data)

            if data is None:
                raise ValueError("fppool_featurizer_returned_none")

            required_attrs = ["x", "edge_index", "edge_attr", "fp", "fp_length"]
            missing = [attr for attr in required_attrs if not hasattr(data, attr)]

            if missing:
                raise ValueError(f"missing_graph_attributes: {missing}")

            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.fp = data.fp.bool()

            graphs.append(data)

        except Exception as error:
            error_rows.append(
                {
                    "row_index": int(i),
                    "smiles": raw_smiles,
                    "label": label if "label" in locals() else None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

        if args.save_every and len(graphs) > 0 and len(graphs) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output, "graphs_checkpoint.pt")
            torch.save(graphs, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    graphs_path = os.path.join(args.output, "graphs.pt")
    errors_path = os.path.join(args.output, "skipped_fppool_errors.csv")
    summary_path = os.path.join(args.output, "graph_summary.json")

    print("\nSaving outputs...")
    torch.save(graphs, graphs_path)

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(errors_path, index=False)

    summary = summarize_graphs(graphs)
    summary.update(
        {
            "input_csv": args.input,
            "smiles_col": smiles_col,
            "label_col": label_col,
            "smarts_col": smarts_col,
            "fp_types": fp_types,
            "num_input_rows": int(len(df)),
            "num_skipped_rows": int(len(error_rows)),
            "graphs_path": graphs_path,
            "errors_path": errors_path,
        }
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Graphs saved to: {graphs_path}")
    print(f"Errors saved to: {errors_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()