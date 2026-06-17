#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Optional, List, Tuple

import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger
except ImportError:
    print("ERROR: RDKit is not installed.")
    print("Install it with: conda install -c conda-forge rdkit")
    sys.exit(1)

RDLogger.DisableLog("rdApp.warning")


SMILES_COLUMNS = [
    "SMILES",
    "smiles",
    "Canonical SMILES",
    "canonical_smiles",
    "PUBCHEM_OPENEYE_CAN_SMILES",
    "PUBCHEM_OPENEYE_ISO_SMILES",
    "PUBCHEM_CACTVS_SMILES",
    "PUBCHEM_EXT_DATASOURCE_SMILES",
]

ACTIVITY_COLUMNS = [
    "PUBCHEM_ACTIVITY_OUTCOME",
    "Activity Outcome",
    "activity_outcome",
    "Outcome",
    "outcome",
    "Activity",
    "activity",
    "Label",
    "label",
]


def parse_skiprows(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def find_column(df: pd.DataFrame, candidates: List[str], user_col: Optional[str]) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(
                f"Column '{user_col}' not found.\nAvailable columns:\n{list(df.columns)}"
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
        f"Available columns:\n{list(df.columns)}\n"
        f"Please use --smiles-col or --activity-col."
    )


def make_label(value) -> Optional[int]:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {"active", "1", "true", "positive", "pos"}:
        return 1

    if text in {"inactive", "0", "false", "negative", "neg"}:
        return 0

    return None


def process_smiles(value) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if pd.isna(value):
        return None, None, "missing_smiles"

    smiles = str(value).strip()

    if smiles == "":
        return None, None, "empty_smiles"

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, None, "invalid_smiles"

        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

        smarts = Chem.MolToSmarts(mol)

        if not canonical_smiles:
            return None, None, "canonical_smiles_failed"

        if not smarts:
            return None, None, "smarts_failed"

        return canonical_smiles, smarts, None

    except Exception as error:
        return None, None, f"rdkit_error: {error}"


def remove_duplicate_conflicts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conflict_smiles = []

    for smiles, group in df.groupby("canonical_smiles"):
        labels = sorted(group["label"].dropna().unique())

        if len(labels) > 1:
            conflict_smiles.append(smiles)

    conflict_df = df[df["canonical_smiles"].isin(conflict_smiles)].copy()
    clean_df = df[~df["canonical_smiles"].isin(conflict_smiles)].copy()

    clean_df = (
        clean_df
        .sort_values("canonical_smiles")
        .drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
    )

    return clean_df, conflict_df


def main():
    parser = argparse.ArgumentParser(
        description="Clean molecular activity data."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="clean_data",
        help="Output folder.",
    )

    parser.add_argument(
        "--skiprows",
        default=None,
        help="Rows to skip, for example: 1,2,3,4",
    )

    parser.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column name.",
    )

    parser.add_argument(
        "--activity-col",
        default=None,
        help="Activity column name.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    skiprows = parse_skiprows(args.skiprows)

    print(f"Input file: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Skipping rows: {skiprows}")

    print("\nLoading CSV...")
    df = pd.read_csv(args.input, skiprows=skiprows, low_memory=False)

    raw_rows = len(df)

    print(f"Rows loaded: {raw_rows}")
    print(f"Columns: {list(df.columns)}")

    smiles_col = find_column(df, SMILES_COLUMNS, args.smiles_col)
    activity_col = find_column(df, ACTIVITY_COLUMNS, args.activity_col)

    print(f"\nUsing SMILES column: {smiles_col}")
    print(f"Using activity column: {activity_col}")

    print("\nCreating labels...")
    df["label"] = df[activity_col].apply(make_label)

    usable_df = df[df["label"].notna()].copy()
    usable_df["label"] = usable_df["label"].astype(int)

    removed_label_df = df[df["label"].isna()].copy()
    removed_label_df.to_csv(
        os.path.join(args.output, "removed_unusable_labels.csv"),
        index=False,
    )

    print(f"Rows with usable labels: {len(usable_df)}")

    print("\nValidating SMILES and creating canonical SMILES + SMARTS...")

    canonical_smiles = []
    smarts = []
    errors = []

    for value in usable_df[smiles_col]:
        can, sma, err = process_smiles(value)
        canonical_smiles.append(can)
        smarts.append(sma)
        errors.append(err)

    usable_df["canonical_smiles"] = canonical_smiles
    usable_df["smarts"] = smarts
    usable_df["smiles_error"] = errors

    valid_df = usable_df[usable_df["smiles_error"].isna()].copy()
    invalid_df = usable_df[usable_df["smiles_error"].notna()].copy()

    invalid_df.to_csv(
        os.path.join(args.output, "removed_invalid_smiles.csv"),
        index=False,
    )

    print(f"Rows with valid SMILES: {len(valid_df)}")
    print(f"Rows removed for invalid SMILES: {len(invalid_df)}")

    print("\nRemoving duplicate molecules and label conflicts...")
    clean_df, conflict_df = remove_duplicate_conflicts(valid_df)

    conflict_df.to_csv(
        os.path.join(args.output, "removed_duplicate_label_conflicts.csv"),
        index=False,
    )

    final_df = clean_df[
        [
            "canonical_smiles",
            "smarts",
            "label",
            smiles_col,
            activity_col,
        ]
    ].copy()

    final_df.to_csv(
        os.path.join(args.output, "cleaned_data.csv"),
        index=False,
    )

    clean_df.to_csv(
        os.path.join(args.output, "cleaned_data_with_original_columns.csv"),
        index=False,
    )

    active = int((final_df["label"] == 1).sum())
    inactive = int((final_df["label"] == 0).sum())

    summary_path = os.path.join(args.output, "cleaning_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Cleaning summary\n")
        f.write("================\n\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"SMILES column: {smiles_col}\n")
        f.write(f"Activity column: {activity_col}\n\n")
        f.write(f"Raw rows: {raw_rows}\n")
        f.write(f"Rows with usable labels: {len(usable_df)}\n")
        f.write(f"Rows with valid SMILES: {len(valid_df)}\n")
        f.write(f"Rows removed for invalid SMILES: {len(invalid_df)}\n")
        f.write(f"Rows removed for duplicate label conflicts: {len(conflict_df)}\n")
        f.write(f"Final clean molecules: {len(final_df)}\n\n")
        f.write(f"Active: {active}\n")
        f.write(f"Inactive: {inactive}\n")

    print("\nDone.")
    print(f"Cleaned data: {args.output}/cleaned_data.csv")
    print(f"Summary: {args.output}/cleaning_summary.txt")
    print(f"Active: {active}")
    print(f"Inactive: {inactive}")


if __name__ == "__main__":
    main()