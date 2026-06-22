#!/usr/bin/env python3

"""
Train a baseline GNN with OPTIONAL molecular fingerprints.

This script supports two modes:

1) Graph-only baseline:
   --fp-type none

2) Graph + fingerprint model:
   --fp-type MorganFP
   --fp-type RdkitFP
   --fp-type MACCSFP
   --fp-type PatternFP
   --fp-type AtomPairFP
   --fp-type TopologicalTorsionFP

For fingerprint mode, the model does this:

    molecular graph -> GNN -> graph_embedding
    SMILES -> RDKit fingerprint -> fp_vector
    concatenate(graph_embedding, fp_vector) -> classifier

So this is a real "GIN + MorganFP" / "GCN + RdkitFP" style baseline.

Important:
- MorganFP/RdkitFP/etc. are computed from each graph's SMILES unless you use --fp-source stored.
- Your PyG Data objects must contain g.smiles or g.original_smiles for computed fingerprints.
- GCN and GIN do not use edge_attr.
- GINE uses edge_attr.
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


# ---------------------------------------------------------------------
# Optional RDKit imports
# ---------------------------------------------------------------------


try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.warning")
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    AllChem = None
    MACCSkeys = None
    DataStructs = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------
# Reproducibility and JSON helpers
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(to_jsonable(metrics), f, indent=2)


# ---------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------


SUPPORTED_FP_TYPES = [
    "none",
    "MorganFP",
    "RdkitFP",
    "MACCSFP",
    "PatternFP",
    "AtomPairFP",
    "TopologicalTorsionFP",
]


def bitvect_to_tensor(bitvect) -> torch.Tensor:
    """
    Convert an RDKit ExplicitBitVect/SparseBitVect into a float tensor.

    Important fix:
    DataStructs.ConvertToNumpyArray expects a NumPy array, not a Python list.
    """
    arr = np.zeros((int(bitvect.GetNumBits()),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return torch.from_numpy(arr).float().view(1, -1)


def get_smiles_from_graph(g) -> Optional[str]:
    """
    Try common attributes that may contain SMILES.
    """
    for attr in ["smiles", "original_smiles", "canonical_smiles"]:
        if hasattr(g, attr):
            value = getattr(g, attr)

            if value is None:
                continue

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], str) and value[0].strip():
                    return value[0].strip()

    return None


def compute_rdkit_fingerprint(
    smiles: str,
    fp_type: str,
    fp_bits: int = 2048,
    morgan_radius: int = 2,
) -> Optional[torch.Tensor]:
    """
    Compute selected RDKit fingerprint from SMILES.

    Returns tensor with shape [1, fp_dim].
    """
    if fp_type == "none":
        return None

    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required to compute fingerprints. "
            "Install with: conda install -c conda-forge rdkit"
        )

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp_type = fp_type.strip()

    if fp_type == "MorganFP":
        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=morgan_radius,
            nBits=fp_bits,
        )

    elif fp_type == "RdkitFP":
        bitvect = Chem.RDKFingerprint(
            mol,
            fpSize=fp_bits,
        )

    elif fp_type == "MACCSFP":
        bitvect = MACCSkeys.GenMACCSKeys(mol)

    elif fp_type == "PatternFP":
        bitvect = Chem.PatternFingerprint(
            mol,
            fpSize=fp_bits,
        )

    elif fp_type == "AtomPairFP":
        bitvect = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=fp_bits,
        )

    elif fp_type == "TopologicalTorsionFP":
        bitvect = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,
            nBits=fp_bits,
        )

    else:
        raise ValueError(
            f"Unsupported fp_type={fp_type}. "
            f"Supported: {SUPPORTED_FP_TYPES}"
        )

    return bitvect_to_tensor(bitvect)


def _tensorize_stored_fp(value) -> Optional[torch.Tensor]:
    """
    Convert a stored fingerprint attribute into shape [1, fp_dim].
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        fp = value.detach().clone().float()
    else:
        fp = torch.tensor(value, dtype=torch.float)

    if fp.numel() == 0:
        return None

    fp = fp.view(1, -1)
    return fp


def get_stored_fingerprint_from_graph(g, fp_type: str) -> Optional[torch.Tensor]:
    """
    Try to find a precomputed fingerprint already stored in the PyG Data object.
    """
    fp_type_lower = fp_type.lower()

    candidate_attrs = [
        "fp",
        "fingerprint",
        "fingerprints",
        "fp_vector",
        "fp_vec",
        "fp_bits",
        "morgan_fp",
        "rdkit_fp",
        "maccs_fp",
        fp_type,
        fp_type_lower,
    ]

    if fp_type == "MorganFP":
        candidate_attrs = [
            "morgan_fp",
            "MorganFP",
            "morgan",
            "fp",
            "fingerprint",
        ] + candidate_attrs
    elif fp_type == "RdkitFP":
        candidate_attrs = [
            "rdkit_fp",
            "RdkitFP",
            "rdkit",
            "fp",
            "fingerprint",
        ] + candidate_attrs
    elif fp_type == "MACCSFP":
        candidate_attrs = [
            "maccs_fp",
            "MACCSFP",
            "maccs",
            "fp",
            "fingerprint",
        ] + candidate_attrs

    seen = set()
    unique_attrs = []

    for attr in candidate_attrs:
        if attr not in seen:
            unique_attrs.append(attr)
            seen.add(attr)

    for attr in unique_attrs:
        if hasattr(g, attr):
            fp = _tensorize_stored_fp(getattr(g, attr))
            if fp is not None:
                return fp

    return None


def attach_fingerprints_to_graphs(
    graphs: List,
    fp_type: str,
    fp_source: str = "compute",
    fp_bits: int = 2048,
    morgan_radius: int = 2,
) -> Tuple[List, Optional[int], Dict[str, Any]]:
    """
    Attach g.fp to every graph.

    fp_type:
      - none: no fingerprint is attached
      - MorganFP/RdkitFP/etc.: attach selected fingerprint

    fp_source:
      - compute: compute from SMILES using RDKit
      - stored: read existing fingerprint attribute from Data object
      - auto: try stored first, then compute from SMILES
    """
    if fp_type == "none":
        return graphs, None, {
            "fp_type": fp_type,
            "fp_source": fp_source,
            "fp_dim": None,
            "computed": 0,
            "loaded_stored": 0,
            "skipped_missing_fp": 0,
        }

    if fp_type not in SUPPORTED_FP_TYPES:
        raise ValueError(f"Unsupported fp_type={fp_type}. Supported: {SUPPORTED_FP_TYPES}")

    if fp_source not in ["compute", "stored", "auto"]:
        raise ValueError("--fp-source must be one of: compute, stored, auto")

    cleaned = []
    skipped = 0
    computed = 0
    loaded_stored = 0
    fp_dim = None

    for idx, g in enumerate(graphs):
        fp = None

        if fp_source in ["stored", "auto"]:
            fp = get_stored_fingerprint_from_graph(g, fp_type)
            if fp is not None:
                loaded_stored += 1

        if fp is None and fp_source in ["compute", "auto"]:
            smiles = get_smiles_from_graph(g)

            if smiles is None:
                print(
                    f"Skipping graph {idx}: cannot compute {fp_type}; "
                    "no smiles/original_smiles/canonical_smiles attribute found."
                )
                skipped += 1
                continue

            fp = compute_rdkit_fingerprint(
                smiles=smiles,
                fp_type=fp_type,
                fp_bits=fp_bits,
                morgan_radius=morgan_radius,
            )

            if fp is None:
                print(f"Skipping graph {idx}: RDKit failed to compute {fp_type} from SMILES.")
                skipped += 1
                continue

            computed += 1

        if fp is None:
            print(f"Skipping graph {idx}: no fingerprint available for fp_type={fp_type}.")
            skipped += 1
            continue

        if fp.dim() != 2 or fp.shape[0] != 1:
            fp = fp.view(1, -1)

        current_fp_dim = int(fp.shape[1])

        if fp_dim is None:
            fp_dim = current_fp_dim
        elif current_fp_dim != fp_dim:
            raise ValueError(
                f"Inconsistent fingerprint dimensions. "
                f"Expected {fp_dim}, got {current_fp_dim} at graph {idx}."
            )

        g.fp = fp.float()
        cleaned.append(g)

    if len(cleaned) == 0:
        raise ValueError(
            f"No graphs left after attaching fingerprints. "
            f"Check that your graphs contain SMILES or stored fingerprint attributes."
        )

    stats = {
        "fp_type": fp_type,
        "fp_source": fp_source,
        "fp_bits": fp_bits,
        "morgan_radius": morgan_radius,
        "fp_dim": fp_dim,
        "computed": computed,
        "loaded_stored": loaded_stored,
        "skipped_missing_fp": skipped,
        "graphs_after_fp_filtering": len(cleaned),
    }

    print("\nFingerprint attachment summary:")
    print(json.dumps(to_jsonable(stats), indent=2))

    return cleaned, fp_dim, stats


# ---------------------------------------------------------------------
# Graph loading and validation
# ---------------------------------------------------------------------


def _has_non_none_attr(data, attr_name: str) -> bool:
    return hasattr(data, attr_name) and getattr(data, attr_name) is not None


def load_graphs(path: str) -> List:
    print(f"Loading graphs from: {path}")
    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError("Expected graphs.pt to contain a list of PyG Data objects.")

    cleaned = []
    skipped = 0

    for idx, g in enumerate(graphs):
        required_attrs = ["x", "edge_index", "y"]
        missing = [a for a in required_attrs if not _has_non_none_attr(g, a)]

        if missing:
            print(f"Skipping graph {idx}: missing required attributes {missing}")
            skipped += 1
            continue

        try:
            g.x = g.x.float()

            if g.x.dim() == 1:
                g.x = g.x.view(-1, 1)

            if g.x.dim() != 2:
                print(f"Skipping graph {idx}: x must be 2D, got shape {tuple(g.x.shape)}")
                skipped += 1
                continue

            if g.x.shape[0] == 0:
                print(f"Skipping graph {idx}: graph has zero nodes.")
                skipped += 1
                continue

            if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
                print(
                    f"Skipping graph {idx}: edge_index must have shape [2, num_edges], "
                    f"got {tuple(g.edge_index.shape)}"
                )
                skipped += 1
                continue

            g.edge_index = g.edge_index.long()

            num_nodes = int(g.x.shape[0])
            num_edges = int(g.edge_index.shape[1])

            if num_edges > 0:
                min_edge_idx = int(g.edge_index.min().item())
                max_edge_idx = int(g.edge_index.max().item())

                if min_edge_idx < 0 or max_edge_idx >= num_nodes:
                    print(
                        f"Skipping graph {idx}: edge_index contains invalid node indices. "
                        f"num_nodes={num_nodes}, min_edge_idx={min_edge_idx}, "
                        f"max_edge_idx={max_edge_idx}"
                    )
                    skipped += 1
                    continue

            if _has_non_none_attr(g, "edge_attr"):
                g.edge_attr = g.edge_attr.float()

                if g.edge_attr.dim() == 1 and g.edge_attr.numel() > 0:
                    g.edge_attr = g.edge_attr.view(-1, 1)

                if g.edge_attr.dim() not in [1, 2]:
                    print(
                        f"Skipping graph {idx}: edge_attr must be 1D or 2D, "
                        f"got shape {tuple(g.edge_attr.shape)}"
                    )
                    skipped += 1
                    continue

                if g.edge_attr.numel() > 0:
                    edge_attr_rows = int(g.edge_attr.shape[0])

                    if edge_attr_rows != num_edges:
                        print(
                            f"Skipping graph {idx}: edge_attr has {edge_attr_rows} rows, "
                            f"but edge_index has {num_edges} edges."
                        )
                        skipped += 1
                        continue

            g.y = g.y.long().view(-1)

            if g.y.numel() != 1:
                print(f"Skipping graph {idx}: expected one label, got {g.y.numel()}")
                skipped += 1
                continue

            label = int(g.y.item())

            if label not in [0, 1]:
                print(f"Skipping graph {idx}: label must be 0 or 1, got {label}")
                skipped += 1
                continue

            cleaned.append(g)

        except Exception as e:
            print(f"Skipping graph {idx} due to error: {e}")
            skipped += 1

    if len(cleaned) == 0:
        raise ValueError("No valid graphs were loaded.")

    print(f"Loaded valid graphs: {len(cleaned)}")
    print(f"Skipped invalid graphs: {skipped}")

    return cleaned


def infer_dimensions_and_validate_graphs(graphs: List) -> Tuple[int, Optional[int]]:
    first = graphs[0]
    node_dim = int(first.x.shape[1])

    for i, g in enumerate(graphs):
        if g.x.dim() != 2:
            raise ValueError(f"Graph {i} has invalid x shape: {tuple(g.x.shape)}")

        if int(g.x.shape[1]) != node_dim:
            raise ValueError(
                f"Graph {i} has node feature dim {g.x.shape[1]}, "
                f"but expected {node_dim}."
            )

        if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
            raise ValueError(
                f"Graph {i} has invalid edge_index shape: {tuple(g.edge_index.shape)}"
            )

    edge_dims = set()

    for g in graphs:
        if _has_non_none_attr(g, "edge_attr"):
            if g.edge_attr.dim() == 2 and g.edge_attr.numel() > 0:
                edge_dims.add(int(g.edge_attr.shape[1]))
            elif g.edge_attr.dim() == 1 and g.edge_attr.numel() > 0:
                edge_dims.add(1)

    if len(edge_dims) == 0:
        edge_dim = None
    elif len(edge_dims) == 1:
        edge_dim = list(edge_dims)[0]
    else:
        raise ValueError(f"Inconsistent edge_attr dimensions found: {edge_dims}")

    filled_missing_edge_attr = 0

    if edge_dim is not None:
        for i, g in enumerate(graphs):
            num_edges = int(g.edge_index.shape[1])

            if not _has_non_none_attr(g, "edge_attr"):
                g.edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float)
                filled_missing_edge_attr += 1
                continue

            if g.edge_attr.dim() == 1:
                if g.edge_attr.numel() == 0:
                    g.edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float)
                    filled_missing_edge_attr += 1
                else:
                    g.edge_attr = g.edge_attr.view(-1, 1)

            if g.edge_attr.dim() != 2:
                raise ValueError(
                    f"Graph {i} has invalid edge_attr shape: {tuple(g.edge_attr.shape)}"
                )

            if int(g.edge_attr.shape[1]) != edge_dim:
                raise ValueError(
                    f"Graph {i} has edge feature dim {g.edge_attr.shape[1]}, "
                    f"but expected {edge_dim}."
                )

            if int(g.edge_attr.shape[0]) != num_edges:
                raise ValueError(
                    f"Graph {i} has {g.edge_attr.shape[0]} edge_attr rows, "
                    f"but edge_index has {num_edges} edges."
                )

    if filled_missing_edge_attr > 0:
        print(
            f"Warning: filled missing edge_attr with zeros for "
            f"{filled_missing_edge_attr} graphs."
        )

    return node_dim, edge_dim


def count_labels(graphs: Sequence) -> Tuple[int, int]:
    labels = torch.tensor([int(g.y.item()) for g in graphs])
    inactive = int((labels == 0).sum().item())
    active = int((labels == 1).sum().item())
    return inactive, active


# ---------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------


def stratified_split(
    graphs: List,
    train_frac: float = 0.72,
    val_frac: float = 0.08,
    seed: int = 42,
) -> Tuple[List, List, List]:
    rng = random.Random(seed)

    inactive = [g for g in graphs if int(g.y.item()) == 0]
    active = [g for g in graphs if int(g.y.item()) == 1]

    if len(inactive) == 0 or len(active) == 0:
        raise ValueError(
            f"Need both classes for stratified split. "
            f"Found inactive={len(inactive)}, active={len(active)}."
        )

    rng.shuffle(inactive)
    rng.shuffle(active)

    def split_class(items: List) -> Tuple[List, List, List]:
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]

        return train, val, test

    inactive_train, inactive_val, inactive_test = split_class(inactive)
    active_train, active_val, active_test = split_class(active)

    train = inactive_train + active_train
    val = inactive_val + active_val
    test = inactive_test + active_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def _safe_graph_attr_for_csv(g, attr_name: str) -> Optional[Any]:
    if not hasattr(g, attr_name):
        return None

    value = getattr(g, attr_name)

    if value is None:
        return None

    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
    except Exception:
        pass

    return value


def save_split_assignments(
    train_graphs: List,
    val_graphs: List,
    test_graphs: List,
    output_path: str,
) -> None:
    rows = []

    for split_name, split_graphs in [
        ("train", train_graphs),
        ("val", val_graphs),
        ("test", test_graphs),
    ]:
        for local_idx, g in enumerate(split_graphs):
            row = {
                "split": split_name,
                "local_index_within_split": local_idx,
                "y": int(g.y.item()),
            }

            for attr in ["row_index", "smiles", "original_smiles", "smarts"]:
                value = _safe_graph_attr_for_csv(g, attr)
                if value is not None:
                    row[attr] = value

            rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def get_pooling(pooling: str):
    if pooling == "mean":
        return global_mean_pool

    if pooling == "sum":
        return global_add_pool

    if pooling == "max":
        return global_max_pool

    raise ValueError(f"Unsupported pooling: {pooling}")


class GNNWithOptionalFP(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int],
        fp_dim: Optional[int],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = "GCN",
        pooling: str = "mean",
        num_classes: int = 2,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.gnn_type = gnn_type.upper()
        self.pooling_name = pooling
        self.pool = get_pooling(pooling)
        self.dropout = dropout
        self.fp_dim = fp_dim

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if self.gnn_type == "GCN":
            self.convs.append(GCNConv(node_dim, hidden_dim))

            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        elif self.gnn_type == "GIN":
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(node_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            )

            for _ in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        )
                    )
                )

        elif self.gnn_type == "GINE":
            if edge_dim is None:
                raise ValueError("GINE requires edge_attr, but edge_dim is None.")

            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(node_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=edge_dim,
                )
            )

            for _ in range(num_layers - 1):
                self.convs.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        edge_dim=edge_dim,
                    )
                )

        else:
            raise ValueError("gnn_type must be one of: GCN, GIN, GINE")

        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        classifier_input_dim = hidden_dim

        if fp_dim is not None and fp_dim > 0:
            classifier_input_dim += fp_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = getattr(data, "edge_attr", None)

        for conv, bn in zip(self.convs, self.batch_norms):
            if self.gnn_type == "GINE":
                if edge_attr is None:
                    raise ValueError("GINE received batch without edge_attr.")
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = self.pool(x, batch)

        if self.fp_dim is not None and self.fp_dim > 0:
            if not hasattr(data, "fp"):
                raise ValueError(
                    "Model expects fingerprint features, but batch has no data.fp."
                )

            fp = data.fp.float()

            if fp.dim() == 1:
                fp = fp.view(data.num_graphs, -1)

            if fp.dim() != 2:
                raise ValueError(f"Expected batch.fp to be 2D, got shape {tuple(fp.shape)}")

            if fp.shape[0] != data.num_graphs:
                fp = fp.view(data.num_graphs, -1)

            if fp.shape[1] != self.fp_dim:
                raise ValueError(
                    f"Expected fp_dim={self.fp_dim}, got batch.fp shape {tuple(fp.shape)}"
                )

            fp = fp.to(graph_embedding.device)
            model_input = torch.cat([graph_embedding, fp], dim=1)

        else:
            model_input = graph_embedding

        logits = self.classifier(model_input)

        return logits


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def compute_basic_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = torch.tensor(y_true).long()
    y_pred = torch.tensor(y_pred).long()

    if y_true.numel() == 0:
        raise ValueError("Cannot compute metrics on an empty set.")

    accuracy = (y_true == y_pred).float().mean().item()

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (recall + specificity) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def add_ranking_metrics(metrics: Dict[str, Any], y_true, y_prob) -> Dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

    except Exception as e:
        print(f"Warning: sklearn metrics failed: {e}")
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def calculate_topk_metrics(
    pred_df: pd.DataFrame,
    overall_active_rate: float,
    k_values: Sequence[int] = (100, 500, 1000),
) -> pd.DataFrame:
    rows = []
    total_actives = int(pred_df["y_true"].sum())

    for k in k_values:
        if k > len(pred_df):
            print(f"Skipping K={k}; only {len(pred_df)} predictions available.")
            continue

        top_k = pred_df.head(k)
        true_actives = int(top_k["y_true"].sum())

        precision_at_k = true_actives / k
        recall_at_k = true_actives / total_actives if total_actives > 0 else 0.0

        ef_at_k = (
            precision_at_k / overall_active_rate
            if overall_active_rate > 0
            else None
        )

        rows.append(
            {
                "k": k,
                "true_actives_in_top_k": true_actives,
                "total_actives": total_actives,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "overall_active_rate": overall_active_rate,
                "ef_at_k": ef_at_k,
            }
        )

    return pd.DataFrame(rows)


def default_threshold_grid() -> List[float]:
    very_low = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075]
    low = [round(i / 1000, 3) for i in range(10, 100)]  # 0.010 to 0.099
    regular = [round(i / 100, 2) for i in range(10, 100)]  # 0.10 to 0.99
    return sorted(set(very_low + low + regular))


def threshold_sweep(
    pred_df: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = default_threshold_grid()

    y_true = pred_df["y_true"].tolist()
    rows = []

    for threshold in thresholds:
        y_pred = (pred_df["prob_active"] >= threshold).astype(int).tolist()
        metrics = compute_basic_metrics(y_true, y_pred)

        rows.append(
            {
                "threshold": threshold,
                "predicted_active": int(sum(y_pred)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def select_threshold_from_validation(
    val_threshold_df: pd.DataFrame,
    metric: str = "f1",
    min_recall: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    df = val_threshold_df.copy()

    if min_recall > 0:
        df = df[df["recall"] >= min_recall].copy()

    if len(df) == 0:
        print(
            f"No threshold satisfied min_recall={min_recall}. "
            "Using all thresholds instead."
        )
        df = val_threshold_df.copy()

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in threshold dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )

    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]

    return float(best_row["threshold"]), best_row.to_dict()


def save_confusion_matrix(metrics: Dict[str, Any], output_path: str) -> pd.DataFrame:
    cm = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        index=["Actual inactive", "Actual active"],
        columns=["Predicted inactive", "Predicted active"],
    )

    cm.to_csv(output_path)
    return cm


def get_model_selection_score(metrics: Dict[str, Any], requested_metric: str) -> Tuple[float, str]:
    score = metrics.get(requested_metric)

    if score is None:
        return float(metrics["balanced_accuracy"]), "balanced_accuracy"

    if isinstance(score, float) and math.isnan(score):
        return float(metrics["balanced_accuracy"]), "balanced_accuracy"

    return float(score), requested_metric


def fmt_metric(value: Any) -> str:
    if value is None:
        return "None"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


# ---------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch)
        y = batch.y.view(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    if total_graphs == 0:
        raise ValueError("Training loader produced zero graphs.")

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.50) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_graphs = 0

    all_true = []
    all_pred = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        y = batch.y.view(-1)

        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
        all_prob.extend(probs.detach().cpu().tolist())

    if total_graphs == 0:
        raise ValueError("Evaluation loader produced zero graphs.")

    metrics = compute_basic_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / total_graphs
    metrics["threshold"] = threshold

    metrics = add_ranking_metrics(metrics, all_true, all_prob)

    return metrics


@torch.no_grad()
def collect_predictions(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]

        preds_default = (probs >= 0.50).long()
        y_true = batch.y.view(-1)

        for i in range(batch.num_graphs):
            row = {
                "y_true": int(y_true[i].detach().cpu().item()),
                "y_pred_threshold_0.50": int(preds_default[i].detach().cpu().item()),
                "prob_active": float(probs[i].detach().cpu().item()),
            }

            if hasattr(batch, "row_index"):
                try:
                    row["row_index"] = int(
                        batch.row_index.view(-1)[i].detach().cpu().item()
                    )
                except Exception:
                    pass

            if hasattr(batch, "smiles") and isinstance(batch.smiles, list):
                try:
                    row["smiles"] = batch.smiles[i]
                except Exception:
                    pass

            if hasattr(batch, "original_smiles") and isinstance(batch.original_smiles, list):
                try:
                    row["original_smiles"] = batch.original_smiles[i]
                except Exception:
                    pass

            if hasattr(batch, "smarts") and isinstance(batch.smarts, list):
                try:
                    row["smarts"] = batch.smarts[i]
                except Exception:
                    pass

            rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No predictions were collected.")

    df = df.sort_values("prob_active", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def save_ranking_outputs(
    model,
    loader,
    device,
    output_dir: str,
    prefix: str,
    overall_active_rate: float,
    selected_threshold: Optional[float] = None,
) -> Dict[str, str]:
    pred_df = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    if selected_threshold is not None:
        pred_df["selected_threshold"] = selected_threshold
        pred_df["y_pred_selected_threshold"] = (
            pred_df["prob_active"] >= selected_threshold
        ).astype(int)

    ranked_predictions_path = os.path.join(
        output_dir,
        f"{prefix}_ranked_predictions.csv",
    )
    pred_df.to_csv(ranked_predictions_path, index=False)

    topk_df = calculate_topk_metrics(
        pred_df=pred_df,
        overall_active_rate=overall_active_rate,
        k_values=(100, 500, 1000),
    )

    topk_metrics_path = os.path.join(
        output_dir,
        f"{prefix}_topk_metrics.csv",
    )
    topk_df.to_csv(topk_metrics_path, index=False)

    threshold_df = threshold_sweep(pred_df)

    threshold_metrics_path = os.path.join(
        output_dir,
        f"{prefix}_threshold_sweep_metrics.csv",
    )
    threshold_df.to_csv(threshold_metrics_path, index=False)

    y_true = pred_df["y_true"].tolist()
    y_pred_default = pred_df["y_pred_threshold_0.50"].tolist()

    default_metrics = compute_basic_metrics(y_true, y_pred_default)
    default_metrics["threshold"] = 0.50

    confusion_matrix_path = os.path.join(
        output_dir,
        f"{prefix}_confusion_matrix_threshold_0.50.csv",
    )
    confusion_matrix_df = save_confusion_matrix(
        default_metrics,
        confusion_matrix_path,
    )

    default_metrics_path = os.path.join(
        output_dir,
        f"{prefix}_metrics_threshold_0.50.json",
    )
    save_metrics_json(default_metrics, default_metrics_path)

    print(f"\n{prefix} confusion matrix at threshold 0.50:")
    print(confusion_matrix_df)

    print(f"\n{prefix} Top-K metrics:")
    print(topk_df)

    print(f"\n{prefix} threshold sweep:")
    print(threshold_df)

    return {
        "ranked_predictions_path": ranked_predictions_path,
        "topk_metrics_path": topk_metrics_path,
        "threshold_metrics_path": threshold_metrics_path,
        "confusion_matrix_threshold_0.50_path": confusion_matrix_path,
        "metrics_threshold_0.50_path": default_metrics_path,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline GNN with optional molecular fingerprints. "
            "Includes Top-K metrics and threshold adjustment analysis."
        )
    )

    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to graphs.pt.",
    )

    parser.add_argument(
        "--output",
        default="baseline_gnn_fp_output",
        help="Output folder.",
    )

    parser.add_argument(
        "--gnn-type",
        default="GCN",
        choices=["GCN", "GIN", "GINE"],
        help="GNN type.",
    )

    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "sum", "max"],
        help="Graph pooling method.",
    )

    parser.add_argument(
        "--fp-type",
        default="none",
        choices=SUPPORTED_FP_TYPES,
        help=(
            "Fingerprint type. Use 'none' for graph-only baseline. "
            "Use MorganFP/RdkitFP/MACCSFP/etc. for graph + fingerprint model."
        ),
    )

    parser.add_argument(
        "--fp-source",
        default="compute",
        choices=["compute", "stored", "auto"],
        help=(
            "How to get fingerprint vectors. "
            "'compute' computes from SMILES using RDKit. "
            "'stored' reads an existing Data attribute. "
            "'auto' tries stored first, then computes from SMILES."
        ),
    )

    parser.add_argument(
        "--fp-bits",
        type=int,
        default=2048,
        help=(
            "Fingerprint bit length for MorganFP, RdkitFP, PatternFP, AtomPairFP, "
            "and TopologicalTorsionFP. Ignored for MACCSFP."
        ),
    )

    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius. Radius 2 is ECFP4-like.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss.",
    )

    parser.add_argument(
        "--model-selection-metric",
        default="average_precision",
        choices=[
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
        ],
        help=(
            "Validation metric used to select the best epoch. "
            "Default: average_precision, useful for imbalanced datasets."
        ),
    )

    parser.add_argument(
        "--threshold-selection-metric",
        default="f1",
        choices=["f1", "precision", "recall", "balanced_accuracy", "specificity"],
        help=(
            "Metric used to select the best classification threshold on the "
            "validation set. Default: f1."
        ),
    )

    parser.add_argument(
        "--min-recall-for-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional minimum recall constraint when selecting threshold. "
            "Example: 0.50 means select best threshold only among thresholds "
            "with validation recall >= 0.50."
        ),
    )

    parser.add_argument(
        "--save-full-dataset-ranking",
        action="store_true",
        help=(
            "Also rank all compounds in the full dataset. "
            "This includes training compounds, so label it clearly."
        ),
    )

    parser.add_argument(
        "--eval-train-metrics",
        action="store_true",
        help=(
            "Also evaluate train metrics every epoch. Useful for overfitting "
            "analysis but slower on large datasets."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"GNN type: {args.gnn_type}")
    print(f"Pooling: {args.pooling}")
    print(f"FP type: {args.fp_type}")
    print(f"FP source: {args.fp_source}")
    print(f"FP bits: {args.fp_bits}")
    print(f"Morgan radius: {args.morgan_radius}")
    print(f"Model-selection metric: {args.model_selection_metric}")
    print(f"Threshold-selection metric: {args.threshold_selection_metric}")
    print(f"Minimum recall for threshold selection: {args.min_recall_for_threshold}")
    print(f"Num workers: {args.num_workers}")

    if args.fp_type != "none" and args.fp_source in ["compute", "auto"] and not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for computed fingerprints. "
            "Install with: conda install -c conda-forge rdkit"
        )

    graphs = load_graphs(args.graphs)

    node_dim, edge_dim = infer_dimensions_and_validate_graphs(graphs)

    graphs, fp_dim, fp_stats = attach_fingerprints_to_graphs(
        graphs=graphs,
        fp_type=args.fp_type,
        fp_source=args.fp_source,
        fp_bits=args.fp_bits,
        morgan_radius=args.morgan_radius,
    )

    inactive, active = count_labels(graphs)

    print(f"Number of graphs after filtering: {len(graphs)}")
    print(f"Inactive: {inactive}")
    print(f"Active: {active}")
    print(f"Overall active rate: {active / len(graphs):.6f}")
    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")
    print(f"Fingerprint dim: {fp_dim}")

    if args.gnn_type == "GINE" and edge_dim is None:
        raise ValueError(
            "You selected GINE, but no edge_attr was found. "
            "Use GCN/GIN or regenerate graphs with edge features."
        )

    if args.gnn_type in ["GCN", "GIN"]:
        print(
            f"Note: {args.gnn_type} ignores edge_attr even if present. "
            "Use GINE if you want the model to use bond/edge features."
        )

    if args.fp_type != "none":
        print(
            f"Model mode: {args.gnn_type} + {args.fp_type}. "
            "The classifier input is concatenate(graph_embedding, fingerprint_vector)."
        )
    else:
        print(f"Model mode: graph-only {args.gnn_type}; no fingerprint vector is used.")

    train_graphs, val_graphs, test_graphs = stratified_split(
        graphs,
        train_frac=0.72,
        val_frac=0.08,
        seed=args.seed,
    )

    print(f"Train graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    train_inactive, train_active = count_labels(train_graphs)
    val_inactive, val_active = count_labels(val_graphs)
    test_inactive, test_active = count_labels(test_graphs)

    print(f"Train inactive/active: {train_inactive}/{train_active}")
    print(f"Val inactive/active: {val_inactive}/{val_active}")
    print(f"Test inactive/active: {test_inactive}/{test_active}")

    if train_inactive == 0 or train_active == 0:
        raise ValueError(
            "Training split must contain both inactive and active samples. "
            f"Found inactive={train_inactive}, active={train_active}."
        )

    split_assignments_path = os.path.join(args.output, "split_assignments.csv")

    save_split_assignments(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        output_path=split_assignments_path,
    )

    print(f"Split assignments saved to: {split_assignments_path}")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GNNWithOptionalFP(
        node_dim=node_dim,
        edge_dim=edge_dim,
        fp_dim=fp_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        pooling=args.pooling,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss.")
        class_weights_json = None

    else:
        total = train_inactive + train_active

        weight_inactive = total / (2 * train_inactive)
        weight_active = total / (2 * train_active)

        class_weights = torch.tensor(
            [weight_inactive, weight_active],
            dtype=torch.float,
            device=device,
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        class_weights_json = {
            "inactive": weight_inactive,
            "active": weight_active,
        }

        print(
            f"Using class weights: "
            f"inactive={weight_inactive:.4f}, active={weight_active:.4f}"
        )

    best_val_score = -1.0
    best_epoch = 0
    best_metric_used = args.model_selection_metric
    history = []

    best_model_path = os.path.join(args.output, "best_model.pt")
    final_model_path = os.path.join(args.output, "final_model.pt")
    history_path = os.path.join(args.output, "training_history.json")
    summary_path = os.path.join(args.output, "summary.json")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.50,
        )

        train_metrics = None

        if args.eval_train_metrics:
            train_metrics = evaluate(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                threshold=0.50,
            )

        val_score, metric_used = get_model_selection_score(
            val_metrics,
            args.model_selection_metric,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "model_selection_metric_requested": args.model_selection_metric,
            "model_selection_metric_used": metric_used,
            "model_selection_score": val_score,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        if train_metrics is not None:
            row.update({f"train_{k}": v for k, v in train_metrics.items()})

        history.append(row)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_metric_used = metric_used

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "fp_dim": fp_dim,
                    "fp_stats": fp_stats,
                    "best_epoch": best_epoch,
                    "model_selection_metric_requested": args.model_selection_metric,
                    "model_selection_metric_used": best_metric_used,
                    "best_val_score": best_val_score,
                    "class_weights": class_weights_json,
                },
                best_model_path,
            )

        train_metric_text = ""

        if train_metrics is not None:
            train_metric_text = (
                f" | train_bal_acc={train_metrics['balanced_accuracy']:.4f}"
                f" | train_f1={train_metrics['f1']:.4f}"
                f" | train_ap={fmt_metric(train_metrics['average_precision'])}"
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}"
            f"{train_metric_text} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_ap={fmt_metric(val_metrics['average_precision'])} | "
            f"val_auc={fmt_metric(val_metrics['roc_auc'])} | "
            f"selection_score={val_score:.4f}({metric_used})"
        )

    print("\nLoading best model for final evaluation...")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nRunning validation threshold sweep...")

    val_pred_df = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
    )

    val_ranked_predictions_path = os.path.join(
        args.output,
        "val_ranked_predictions.csv",
    )
    val_pred_df.to_csv(val_ranked_predictions_path, index=False)

    val_threshold_df = threshold_sweep(val_pred_df)

    val_threshold_sweep_path = os.path.join(
        args.output,
        "val_threshold_sweep_metrics.csv",
    )
    val_threshold_df.to_csv(val_threshold_sweep_path, index=False)

    selected_threshold, selected_val_row = select_threshold_from_validation(
        val_threshold_df,
        metric=args.threshold_selection_metric,
        min_recall=args.min_recall_for_threshold,
    )

    selected_threshold_info_path = os.path.join(
        args.output,
        "selected_threshold_from_validation.json",
    )

    selected_threshold_info = {
        "selection_metric": args.threshold_selection_metric,
        "min_recall_for_threshold": args.min_recall_for_threshold,
        "selected_threshold": selected_threshold,
        "selected_validation_row": selected_val_row,
        "note": (
            "Threshold was selected using validation set only, then applied to test set."
        ),
    }

    save_metrics_json(selected_threshold_info, selected_threshold_info_path)

    print("\nSelected threshold from validation set:")
    print(json.dumps(to_jsonable(selected_threshold_info), indent=2))

    print("\nEvaluating test set at default threshold 0.50...")

    test_metrics_default = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=0.50,
    )

    test_metrics_default_path = os.path.join(
        args.output,
        "test_metrics_default_threshold_0.50.json",
    )
    save_metrics_json(test_metrics_default, test_metrics_default_path)

    test_confusion_matrix_default_path = os.path.join(
        args.output,
        "test_confusion_matrix_threshold_0.50_from_evaluate.csv",
    )
    test_confusion_matrix_default_df = save_confusion_matrix(
        test_metrics_default,
        test_confusion_matrix_default_path,
    )

    print("\nTest metrics using default threshold 0.50:")
    print(json.dumps(to_jsonable(test_metrics_default), indent=2))

    print("\nTest confusion matrix using default threshold 0.50:")
    print(test_confusion_matrix_default_df)

    print("\nApplying selected threshold to test set...")

    test_metrics_selected_threshold = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=selected_threshold,
    )

    selected_test_metrics_path = os.path.join(
        args.output,
        "test_metrics_selected_threshold.json",
    )
    save_metrics_json(test_metrics_selected_threshold, selected_test_metrics_path)

    selected_test_confusion_matrix_path = os.path.join(
        args.output,
        "test_confusion_matrix_selected_threshold.csv",
    )
    selected_test_confusion_matrix_df = save_confusion_matrix(
        test_metrics_selected_threshold,
        selected_test_confusion_matrix_path,
    )

    print("\nTest metrics using selected threshold:")
    print(json.dumps(to_jsonable(test_metrics_selected_threshold), indent=2))

    print("\nTest confusion matrix using selected threshold:")
    print(selected_test_confusion_matrix_df)

    print("\nSaving test ranking outputs...")

    test_active_rate = test_active / len(test_graphs)

    test_output_paths = save_ranking_outputs(
        model=model,
        loader=test_loader,
        device=device,
        output_dir=args.output,
        prefix="test",
        overall_active_rate=test_active_rate,
        selected_threshold=selected_threshold,
    )

    full_dataset_output_paths = None

    if args.save_full_dataset_ranking:
        print("\nSaving full-dataset ranking outputs...")

        all_loader = DataLoader(
            graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        full_active_rate = active / len(graphs)

        full_dataset_output_paths = save_ranking_outputs(
            model=model,
            loader=all_loader,
            device=device,
            output_dir=args.output,
            prefix="full_dataset",
            overall_active_rate=full_active_rate,
            selected_threshold=selected_threshold,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "fp_dim": fp_dim,
            "fp_stats": fp_stats,
            "best_epoch": best_epoch,
            "model_selection_metric_requested": args.model_selection_metric,
            "model_selection_metric_used": best_metric_used,
            "best_val_score": best_val_score,
            "class_weights": class_weights_json,
            "test_metrics_default_threshold_0.50": test_metrics_default,
            "test_metrics_selected_threshold": test_metrics_selected_threshold,
            "selected_threshold": selected_threshold,
        },
        final_model_path,
    )

    with open(history_path, "w") as f:
        json.dump(to_jsonable(history), f, indent=2)

    summary = {
        "graphs": args.graphs,
        "gnn_type": args.gnn_type,
        "pooling": args.pooling,
        "fp_type": args.fp_type,
        "fp_source": args.fp_source,
        "fp_bits": args.fp_bits,
        "morgan_radius": args.morgan_radius,
        "fp_dim": fp_dim,
        "fp_stats": fp_stats,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_graphs": len(graphs),
        "inactive": inactive,
        "active": active,
        "overall_active_rate_full_dataset": active / len(graphs),
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "train_inactive": train_inactive,
        "train_active": train_active,
        "val_inactive": val_inactive,
        "val_active": val_active,
        "test_inactive": test_inactive,
        "test_active": test_active,
        "overall_active_rate_test_set": test_active_rate,
        "class_weights": class_weights_json,
        "best_epoch": best_epoch,
        "model_selection_metric_requested": args.model_selection_metric,
        "model_selection_metric_used": best_metric_used,
        "best_val_score": best_val_score,
        "test_metrics_default_threshold_0.50": test_metrics_default,
        "test_metrics_default_path": test_metrics_default_path,
        "test_confusion_matrix_default_path": test_confusion_matrix_default_path,
        "selected_threshold": selected_threshold,
        "selected_threshold_info_path": selected_threshold_info_path,
        "test_metrics_selected_threshold": test_metrics_selected_threshold,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "history_path": history_path,
        "split_assignments_path": split_assignments_path,
        "val_ranked_predictions_path": val_ranked_predictions_path,
        "val_threshold_sweep_path": val_threshold_sweep_path,
        "selected_test_metrics_path": selected_test_metrics_path,
        "selected_test_confusion_matrix_path": selected_test_confusion_matrix_path,
        "test_output_paths": test_output_paths,
        "full_dataset_output_paths": full_dataset_output_paths,
        "model_description": (
            "This model encodes molecular graph structure with a GNN. "
            "If fp_type is not none, it concatenates the graph embedding with "
            "the selected fingerprint vector before classification."
        ),
    }

    with open(summary_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print("\nDone.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nImportant output files:")
    print(f"Split assignments: {split_assignments_path}")
    print(f"Validation ranked predictions: {val_ranked_predictions_path}")
    print(f"Validation threshold sweep: {val_threshold_sweep_path}")
    print(f"Selected threshold info: {selected_threshold_info_path}")
    print(f"Default-threshold test metrics: {test_metrics_default_path}")
    print(f"Default-threshold test confusion matrix: {test_confusion_matrix_default_path}")
    print(f"Selected-threshold test metrics: {selected_test_metrics_path}")
    print(f"Selected-threshold confusion matrix: {selected_test_confusion_matrix_path}")
    print(f"Test ranked predictions: {test_output_paths['ranked_predictions_path']}")
    print(f"Test Top-K metrics: {test_output_paths['topk_metrics_path']}")
    print(f"Test threshold sweep: {test_output_paths['threshold_metrics_path']}")

    if full_dataset_output_paths is not None:
        print(f"Full-dataset ranked predictions: {full_dataset_output_paths['ranked_predictions_path']}")
        print(f"Full-dataset Top-K metrics: {full_dataset_output_paths['topk_metrics_path']}")
        print(f"Full-dataset threshold sweep: {full_dataset_output_paths['threshold_metrics_path']}")


if __name__ == "__main__":
    main()