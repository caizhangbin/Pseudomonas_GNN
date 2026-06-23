#!/usr/bin/env python3

import argparse
import os
import json

import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def make_morgan_fp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )

    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)

    return torch.tensor(arr, dtype=torch.float).view(1, -1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FPPool-style graphs into graphs with fixed-size molecular MorganFP in data.fp."
    )

    parser.add_argument(
        "--input-graphs",
        required=True,
        help="Input graphs.pt, e.g. AID_1527_fppool_morgan/graphs.pt",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder, e.g. AID_1527_morgan_fixed",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius. Default: 2",
    )

    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length. Default: 2048",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading graphs from: {args.input_graphs}")
    graphs = torch.load(args.input_graphs, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError("Expected input graphs.pt to contain a list of PyG Data objects.")

    converted = []
    skipped = []

    for i, g in enumerate(graphs):
        try:
            if not hasattr(g, "smiles"):
                raise ValueError("graph has no smiles attribute")

            # Keep a copy of the original FPPool-style fp in case you want to inspect it later.
            if hasattr(g, "fp"):
                g.fppool_raw_fp = g.fp

            if hasattr(g, "fp_length"):
                g.fppool_raw_fp_length = g.fp_length

            # Replace data.fp with a fixed molecule-level Morgan fingerprint.
            g.fp = make_morgan_fp_from_smiles(
                g.smiles,
                radius=args.radius,
                n_bits=args.n_bits,
            )

            g.fp_name = f"MorganFP_radius{args.radius}_nBits{args.n_bits}"

            converted.append(g)

            if len(converted) % 10000 == 0:
                print(f"Converted {len(converted)} graphs...")

        except Exception as e:
            skipped.append(
                {
                    "graph_index": i,
                    "smiles": getattr(g, "smiles", None),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    output_graphs_path = os.path.join(args.output_dir, "graphs.pt")
    summary_path = os.path.join(args.output_dir, "fixed_morgan_summary.json")

    torch.save(converted, output_graphs_path)

    summary = {
        "input_graphs": args.input_graphs,
        "output_graphs": output_graphs_path,
        "num_input_graphs": len(graphs),
        "num_converted_graphs": len(converted),
        "num_skipped_graphs": len(skipped),
        "radius": args.radius,
        "n_bits": args.n_bits,
        "fp_shape_first_graph": list(converted[0].fp.shape) if converted else None,
        "note": (
            "data.fp was replaced with a fixed molecule-level Morgan fingerprint. "
            "Original FPPool-style fp was copied to data.fppool_raw_fp."
        ),
        "skipped_examples": skipped[:20],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Converted graphs saved to: {output_graphs_path}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()