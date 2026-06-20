#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

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


def split_smarts_cell(value, delimiter: str) -> List[str]:
    """Return one or more SMARTS patterns from a cell.

    By default, this treats each cell as one SMARTS pattern unless the cell
    contains the delimiter. Semicolon is safer than comma because commas can
    occur inside valid SMARTS atom expressions, e.g. [N,O].
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text == "":
        return []

    if delimiter:
        pieces = [x.strip() for x in text.split(delimiter)]
    else:
        pieces = [text]

    return [x for x in pieces if x]


def read_smarts_file(path: str, delimiter: str) -> List[Tuple[str, str]]:
    """Read SMARTS patterns from a CSV/TSV/TXT file.

    Supported formats:
      1) CSV/TSV with a 'smarts' column and optional 'name' column.
      2) TXT file with one SMARTS pattern per line.
      3) TXT file with name<TAB>SMARTS per line.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMARTS file not found: {path}")

    entries: List[Tuple[str, str]] = []
    lower = path.lower()

    if lower.endswith((".csv", ".tsv")):
        sep = "\t" if lower.endswith(".tsv") else ","
        sdf = pd.read_csv(path, sep=sep, low_memory=False)
        smarts_col = find_column(
            sdf,
            candidates=["smarts", "SMARTS", "pattern", "Pattern"],
            user_col=None,
        )

        name_col = None
        for candidate in ["name", "Name", "pattern_name", "Pattern Name", "id", "ID"]:
            if candidate in sdf.columns:
                name_col = candidate
                break

        for idx, row in sdf.iterrows():
            raw_name = row[name_col] if name_col is not None else f"SMARTS_{idx:05d}"
            name = str(raw_name).strip() if not pd.isna(raw_name) else f"SMARTS_{idx:05d}"
            for smarts in split_smarts_cell(row[smarts_col], delimiter=delimiter):
                entries.append((name, smarts))

    else:
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "\t" in line:
                    name, smarts_text = line.split("\t", 1)
                    name = name.strip() or f"SMARTS_{idx:05d}"
                else:
                    name = f"SMARTS_{idx:05d}"
                    smarts_text = line

                for smarts in split_smarts_cell(smarts_text, delimiter=delimiter):
                    entries.append((name, smarts))

    return entries


def get_smarts_entries(
    df: pd.DataFrame,
    smarts_col: Optional[str],
    smarts_file: Optional[str],
    delimiter: str,
    min_frequency: int,
    max_patterns: Optional[int],
    add_whole_molecule_bit: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Build and validate a fixed SMARTS vocabulary.

    The returned list defines the columns of data.fp. Every molecule will be
    matched against every valid SMARTS pattern in this vocabulary.
    """
    if smarts_file:
        raw_entries = read_smarts_file(smarts_file, delimiter=delimiter)
    else:
        if smarts_col is None:
            raise ValueError(
                "SMARTS-based output requires a SMARTS source. Provide either "
                "--smarts-col or --smarts-file."
            )

        raw_entries = []
        for _, row in df.iterrows():
            for smarts in split_smarts_cell(row[smarts_col], delimiter=delimiter):
                raw_entries.append((smarts, smarts))

    counts = Counter(smarts for _, smarts in raw_entries)

    # Preserve first-seen name for each unique SMARTS string.
    first_name_by_smarts: OrderedDict[str, str] = OrderedDict()
    for name, smarts in raw_entries:
        if smarts not in first_name_by_smarts:
            first_name_by_smarts[smarts] = name

    unique_smarts = [
        smarts for smarts in first_name_by_smarts.keys()
        if counts[smarts] >= min_frequency
    ]

    # If a maximum is requested, keep the most frequent patterns first.
    if max_patterns is not None:
        unique_smarts = sorted(unique_smarts, key=lambda x: (-counts[x], x))[:max_patterns]

    valid_entries: List[Dict] = []
    invalid_entries: List[Dict] = []

    for idx, smarts in enumerate(unique_smarts):
        query = Chem.MolFromSmarts(smarts)
        name = str(first_name_by_smarts[smarts]).strip() or f"SMARTS_{idx:05d}"

        if query is None:
            invalid_entries.append(
                {
                    "name": name,
                    "smarts": smarts,
                    "frequency": int(counts[smarts]),
                    "error": "Chem.MolFromSmarts returned None",
                }
            )
            continue

        valid_entries.append(
            {
                "index": len(valid_entries),
                "name": name,
                "smarts": smarts,
                "frequency": int(counts[smarts]),
                "query": query,
            }
        )

    if add_whole_molecule_bit:
        valid_entries.append(
            {
                "index": len(valid_entries),
                "name": "WHOLE_MOLECULE",
                "smarts": "*",
                "frequency": int(len(df)),
                "query": None,
                "whole_molecule_bit": True,
            }
        )

    if len(valid_entries) == 0:
        raise ValueError("No valid SMARTS patterns found.")

    # Re-number after invalid patterns are removed.
    for idx, entry in enumerate(valid_entries):
        entry["index"] = idx

    return valid_entries, invalid_entries


def smarts_substructure_fp(
    mol,
    smarts_entries: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create atom-by-SMARTS substructure membership matrix.

    fp[a, p] is True when atom a belongs to at least one match of SMARTS
    pattern p. This shape is compatible with FPPool-style downstream training:
    [num_atoms, num_smarts_patterns].
    """
    num_atoms = int(mol.GetNumAtoms())
    num_patterns = len(smarts_entries)

    fp = torch.zeros((num_atoms, num_patterns), dtype=torch.bool)
    present = torch.zeros((num_patterns,), dtype=torch.bool)
    counts = torch.zeros((num_patterns,), dtype=torch.long)

    for pattern_idx, entry in enumerate(smarts_entries):
        if entry.get("whole_molecule_bit", False):
            if num_atoms > 0:
                fp[:, pattern_idx] = True
                present[pattern_idx] = True
                counts[pattern_idx] = 1
            continue

        query = entry["query"]
        matches = mol.GetSubstructMatches(query, uniquify=True)
        counts[pattern_idx] = len(matches)

        if len(matches) == 0:
            continue

        present[pattern_idx] = True
        for match in matches:
            for atom_idx in match:
                fp[int(atom_idx), pattern_idx] = True

    return fp, present, counts


def make_data_object(
    smiles: str,
    label: int,
    row_index: int,
    original_smiles: Optional[str] = None,
) -> Data:
    data = Data()

    # FPPool featurizer expects data.smiles.
    data.smiles = smiles

    # Classification label.
    data.y = torch.tensor([int(label)], dtype=torch.long)

    # Useful metadata.
    data.row_index = torch.tensor([int(row_index)], dtype=torch.long)

    if original_smiles is not None:
        data.original_smiles = str(original_smiles)

    return data


def as_1d_long_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long).view(-1)


def summarize_graphs(graphs: List[Data]) -> dict:
    if not graphs:
        return {}

    labels = torch.cat([g.y for g in graphs]).view(-1)
    active = int((labels == 1).sum().item())
    inactive = int((labels == 0).sum().item())

    first = graphs[0]
    num_with_any_smarts_match = 0
    total_smarts_matches = 0

    for g in graphs:
        if hasattr(g, "smarts_match_present") and bool(g.smarts_match_present.any().item()):
            num_with_any_smarts_match += 1
        if hasattr(g, "num_smarts_matches"):
            total_smarts_matches += int(g.num_smarts_matches.item())

    summary = {
        "num_graphs": len(graphs),
        "active": active,
        "inactive": inactive,
        "node_feature_dim": int(first.x.shape[1]) if hasattr(first, "x") else None,
        "edge_feature_dim": int(first.edge_attr.shape[1]) if hasattr(first, "edge_attr") and first.edge_attr.numel() > 0 else None,
        "fp_shape_first_graph": list(first.fp.shape) if hasattr(first, "fp") else None,
        "fp_length_first_graph": first.fp_length.tolist() if hasattr(first, "fp_length") else None,
        "num_graphs_with_any_smarts_match": num_with_any_smarts_match,
        "total_smarts_matches": total_smarts_matches,
        "example_smiles": first.smiles if hasattr(first, "smiles") else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create PyTorch Geometric graphs with SMARTS-based substructure "
            "identification for downstream GNN + FPPool training."
        )
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
        default="fppool_smarts_processed",
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
        help=(
            "Column containing SMARTS patterns. If --smarts-file is not used, "
            "all valid unique patterns in this column become the fixed SMARTS vocabulary."
        ),
    )

    parser.add_argument(
        "--smarts-file",
        default=None,
        help=(
            "Optional SMARTS vocabulary file. Supports CSV/TSV with a smarts column "
            "and optional name column, or TXT with one SMARTS per line."
        ),
    )

    parser.add_argument(
        "--smarts-delimiter",
        default=";",
        help=(
            "Delimiter for multiple SMARTS patterns in one cell. Default ';'. "
            "Use '' to treat each cell as one pattern. Avoid comma because comma can be valid SMARTS syntax."
        ),
    )

    parser.add_argument(
        "--min-smarts-frequency",
        type=int,
        default=1,
        help="Keep SMARTS patterns observed at least this many times. Default: 1.",
    )

    parser.add_argument(
        "--max-smarts-patterns",
        type=int,
        default=None,
        help="Optional maximum number of SMARTS patterns to keep, ranked by frequency.",
    )

    parser.add_argument(
        "--add-whole-molecule-bit",
        action="store_true",
        help=(
            "Append one extra fingerprint column that is True for all atoms. "
            "This can help avoid completely empty SMARTS fingerprints."
        ),
    )

    parser.add_argument(
        "--fp-source",
        default="smarts",
        choices=["smarts", "concat", "fppool"],
        help=(
            "Which fingerprint matrix to save as data.fp. "
            "'smarts' replaces FPPool fingerprints with SMARTS atom-membership features. "
            "'concat' appends SMARTS features to the original FPPool fingerprints. "
            "'fppool' keeps original FPPool fingerprints but still saves SMARTS match metadata."
        ),
    )

    parser.add_argument(
        "--skip-no-smarts-match",
        action="store_true",
        help="Skip molecules that do not match any SMARTS pattern.",
    )

    parser.add_argument(
        "--fp-types",
        default="RdkitFP",
        help=(
            "Comma-separated FPPool fingerprint types used by the official featurizer. "
            "Needed to generate graph attributes and, if --fp-source concat/fppool, original fingerprints. "
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

    if args.min_smarts_frequency < 1:
        raise ValueError("--min-smarts-frequency must be >= 1")

    fp_types = parse_fp_types(args.fp_types)

    print(f"Input CSV: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"FPPool path: {args.fppool_path}")
    print(f"FPPool fingerprint types: {fp_types}")
    print(f"Fingerprint source saved in data.fp: {args.fp_source}")

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

    if args.smarts_file:
        smarts_col = None
    elif args.smarts_col:
        smarts_col = find_column(df, candidates=["smarts", "SMARTS"], user_col=args.smarts_col)
    else:
        smarts_col = "smarts" if "smarts" in df.columns else None

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Using first {len(df)} rows because --limit was set.")

    print(f"Using SMILES column: {smiles_col}")
    print(f"Using label column: {label_col}")
    print(f"Using SMARTS column: {smarts_col}")
    print(f"Using SMARTS file: {args.smarts_file}")

    print("\nBuilding SMARTS vocabulary...")
    smarts_entries, invalid_smarts_entries = get_smarts_entries(
        df=df,
        smarts_col=smarts_col,
        smarts_file=args.smarts_file,
        delimiter=args.smarts_delimiter,
        min_frequency=args.min_smarts_frequency,
        max_patterns=args.max_smarts_patterns,
        add_whole_molecule_bit=args.add_whole_molecule_bit,
    )

    print(f"Valid SMARTS patterns: {len(smarts_entries)}")
    print(f"Invalid SMARTS patterns skipped: {len(invalid_smarts_entries)}")

    smarts_vocab_path = os.path.join(args.output, "smarts_vocabulary.json")
    invalid_smarts_path = os.path.join(args.output, "invalid_smarts_patterns.csv")

    serializable_vocab = [
        {k: v for k, v in entry.items() if k != "query"}
        for entry in smarts_entries
    ]
    with open(smarts_vocab_path, "w") as f:
        json.dump(serializable_vocab, f, indent=2)

    pd.DataFrame(invalid_smarts_entries).to_csv(invalid_smarts_path, index=False)

    graphs = []
    error_rows = []
    no_smarts_match_rows = []

    print("\nCreating graphs with SMARTS-based substructure fingerprints...")

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

            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                raise ValueError("canonical_smiles_could_not_be_reparsed")

            data = make_data_object(
                smiles=canonical_smiles,
                label=label,
                row_index=int(i),
                original_smiles=raw_smiles,
            )

            # Use the official FPPool featurizer to create molecular graph attributes.
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

            original_fp = data.fp.bool()
            original_fp_length = as_1d_long_tensor(data.fp_length)

            smarts_fp, smarts_present, smarts_counts = smarts_substructure_fp(
                mol=mol,
                smarts_entries=smarts_entries,
            )

            if smarts_fp.shape[0] != data.x.shape[0]:
                raise ValueError(
                    "atom_count_mismatch_between_rdkit_and_fppool: "
                    f"rdkit={smarts_fp.shape[0]}, fppool={data.x.shape[0]}"
                )

            num_smarts_matches = int(smarts_counts.sum().item())
            if num_smarts_matches == 0:
                no_smarts_match_rows.append(
                    {
                        "row_index": int(i),
                        "smiles": canonical_smiles,
                        "label": label,
                    }
                )
                if args.skip_no_smarts_match:
                    raise ValueError("no_smarts_substructure_match")

            # Save SMARTS match metadata on the graph object.
            data.smarts_match_present = smarts_present.bool()
            data.smarts_match_counts = smarts_counts.long()
            data.num_smarts_matches = torch.tensor([num_smarts_matches], dtype=torch.long)

            # Replace, concatenate, or keep the FPPool fingerprint matrix.
            if args.fp_source == "smarts":
                data.fp = smarts_fp.bool()
                data.fp_length = torch.tensor([smarts_fp.shape[1]], dtype=torch.long)
            elif args.fp_source == "concat":
                if original_fp.shape[0] != smarts_fp.shape[0]:
                    raise ValueError(
                        "cannot_concat_fp_due_to_atom_count_mismatch: "
                        f"original={original_fp.shape}, smarts={smarts_fp.shape}"
                    )
                data.fp = torch.cat([original_fp, smarts_fp.bool()], dim=1)
                data.fp_length = torch.cat(
                    [
                        original_fp_length,
                        torch.tensor([smarts_fp.shape[1]], dtype=torch.long),
                    ],
                    dim=0,
                )
            elif args.fp_source == "fppool":
                data.fp = original_fp
                data.fp_length = original_fp_length
            else:
                raise ValueError(f"Unsupported fp_source: {args.fp_source}")

            data.fp = data.fp.bool()
            data.fp_length = as_1d_long_tensor(data.fp_length)

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
    errors_path = os.path.join(args.output, "skipped_fppool_smarts_errors.csv")
    no_smarts_match_path = os.path.join(args.output, "no_smarts_match_rows.csv")
    summary_path = os.path.join(args.output, "graph_summary.json")

    print("\nSaving outputs...")
    torch.save(graphs, graphs_path)

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(errors_path, index=False)

    no_smarts_match_df = pd.DataFrame(no_smarts_match_rows)
    no_smarts_match_df.to_csv(no_smarts_match_path, index=False)

    summary = summarize_graphs(graphs)
    summary.update(
        {
            "input_csv": args.input,
            "smiles_col": smiles_col,
            "label_col": label_col,
            "smarts_col": smarts_col,
            "smarts_file": args.smarts_file,
            "smarts_vocabulary_path": smarts_vocab_path,
            "invalid_smarts_path": invalid_smarts_path,
            "num_smarts_patterns": int(len(smarts_entries)),
            "num_invalid_smarts_patterns": int(len(invalid_smarts_entries)),
            "fp_source": args.fp_source,
            "fppool_fp_types": fp_types,
            "min_smarts_frequency": int(args.min_smarts_frequency),
            "max_smarts_patterns": args.max_smarts_patterns,
            "add_whole_molecule_bit": bool(args.add_whole_molecule_bit),
            "num_input_rows": int(len(df)),
            "num_skipped_rows": int(len(error_rows)),
            "num_no_smarts_match_rows": int(len(no_smarts_match_rows)),
            "skip_no_smarts_match": bool(args.skip_no_smarts_match),
            "graphs_path": graphs_path,
            "errors_path": errors_path,
            "no_smarts_match_path": no_smarts_match_path,
        }
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Graphs saved to: {graphs_path}")
    print(f"Errors saved to: {errors_path}")
    print(f"Rows with no SMARTS match saved to: {no_smarts_match_path}")
    print(f"SMARTS vocabulary saved to: {smarts_vocab_path}")
    print(f"Invalid SMARTS patterns saved to: {invalid_smarts_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
