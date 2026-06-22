#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(obj):
    """
    Convert numpy/pandas/torch scalar values into JSON-safe Python objects.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
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


def save_metrics_json(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(to_jsonable(metrics), f, indent=2)


def load_graphs(path: str):
    print(f"Loading graphs from: {path}")
    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError("Expected graphs.pt to contain a list of PyG Data objects.")

    cleaned = []
    skipped = 0

    for idx, g in enumerate(graphs):
        if not hasattr(g, "x") or not hasattr(g, "edge_index") or not hasattr(g, "y"):
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

            if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
                print(
                    f"Skipping graph {idx}: edge_index must have shape [2, num_edges], "
                    f"got {tuple(g.edge_index.shape)}"
                )
                skipped += 1
                continue

            if hasattr(g, "edge_attr") and g.edge_attr is not None:
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

            g.y = g.y.long().view(-1)

            if g.y.numel() != 1:
                print(f"Skipping graph {idx}: expected one label, got {g.y.numel()}")
                skipped += 1
                continue

            cleaned.append(g)

        except Exception as e:
            print(f"Skipping graph {idx} due to error: {e}")
            skipped += 1

    if len(cleaned) == 0:
        raise ValueError("No valid graphs were loaded.")

    print(f"Loaded graphs: {len(cleaned)}")
    print(f"Skipped invalid graphs: {skipped}")

    return cleaned


def infer_dimensions_and_validate_graphs(graphs: List):
    """
    Infer node_dim and edge_dim safely.

    If some graphs have edge_attr and others do not, this function fills missing
    edge_attr with zeros so PyG batching is consistent. This is useful for GCN/GIN
    and allows GINE to run if edge features exist in the dataset.
    """
    first = graphs[0]
    node_dim = first.x.shape[1]

    for i, g in enumerate(graphs):
        if g.x.dim() != 2:
            raise ValueError(f"Graph {i} has invalid x shape: {tuple(g.x.shape)}")

        if g.x.shape[1] != node_dim:
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
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            if g.edge_attr.dim() == 2:
                edge_dims.add(g.edge_attr.shape[1])
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

            if not hasattr(g, "edge_attr") or g.edge_attr is None:
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

            if g.edge_attr.shape[1] != edge_dim:
                raise ValueError(
                    f"Graph {i} has edge feature dim {g.edge_attr.shape[1]}, "
                    f"but expected {edge_dim}."
                )

            if g.edge_attr.shape[0] != num_edges:
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


def count_labels(graphs):
    labels = torch.tensor([int(g.y.item()) for g in graphs])
    inactive = int((labels == 0).sum().item())
    active = int((labels == 1).sum().item())
    return inactive, active


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

    def split_class(items):
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]

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


def get_pooling(pooling: str):
    if pooling == "mean":
        return global_mean_pool
    elif pooling == "sum":
        return global_add_pool
    elif pooling == "max":
        return global_max_pool
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")


class BaselineGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = "GCN",
        pooling: str = "mean",
        num_classes: int = 2,
    ):
        super().__init__()

        self.gnn_type = gnn_type.upper()
        self.pooling_name = pooling
        self.pool = get_pooling(pooling)
        self.dropout = dropout

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

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        logits = self.classifier(graph_embedding)

        return logits


def compute_basic_metrics(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        y = batch.y.view(-1)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.50):
    """
    Evaluate model using a specified probability threshold.

    threshold = 0.50 means:
      prob_active >= 0.50 -> predicted active
      prob_active < 0.50 -> predicted inactive
    """
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

        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())

    metrics = compute_basic_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / total_graphs
    metrics["threshold"] = threshold

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(set(all_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(all_true, all_prob)
            metrics["average_precision"] = average_precision_score(all_true, all_prob)
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

    except Exception as e:
        print(f"Warning: sklearn metrics failed: {e}")
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Save y_true and prob_active for ranking and threshold analysis.
    """
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
                "y_true": int(y_true[i].cpu().item()),
                "y_pred_threshold_0.50": int(preds_default[i].cpu().item()),
                "prob_active": float(probs[i].cpu().item()),
            }

            if hasattr(batch, "row_index"):
                try:
                    row["row_index"] = int(batch.row_index.view(-1)[i].cpu().item())
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

            rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values("prob_active", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def calculate_topk_metrics(pred_df, overall_active_rate, k_values=(100, 500, 1000)):
    """
    Precision@K = number of true actives in top K / K
    Recall@K = number of true actives in top K / total true actives
    EF@K = Precision@K / overall active rate
    """
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
                "K": k,
                "true_actives_in_top_K": true_actives,
                "total_actives": total_actives,
                f"Precision@{k}": precision_at_k,
                f"Recall@{k}": recall_at_k,
                "overall_active_rate": overall_active_rate,
                f"EF@{k}": ef_at_k,
            }
        )

    return pd.DataFrame(rows)


def default_threshold_grid():
    """
    More useful threshold grid for imbalanced datasets.

    The active rate is often very low, so include thresholds below 0.05.
    """
    low_thresholds = [0.001, 0.002, 0.005, 0.01, 0.02]
    regular_thresholds = [round(i / 100, 2) for i in range(3, 100, 2)]

    thresholds = sorted(set(low_thresholds + regular_thresholds))

    return thresholds


def threshold_sweep(pred_df, thresholds=None):
    """
    Calculate precision, recall, F1, balanced accuracy, specificity,
    and confusion matrix values across probability thresholds.
    """
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
    val_threshold_df,
    metric: str = "f1",
    min_recall: float = 0.0,
):
    """
    Select threshold using validation set only.

    Default: choose threshold with highest validation F1.

    Example:
      --threshold-selection-metric precision
      --min-recall-for-threshold 0.50

    This selects the threshold with highest precision among thresholds where
    validation recall >= 0.50.
    """
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


def save_confusion_matrix(metrics, output_path):
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


def save_ranking_outputs(
    model,
    loader,
    device,
    output_dir,
    prefix,
    overall_active_rate,
    selected_threshold: Optional[float] = None,
):
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


def get_model_selection_score(metrics: Dict[str, Any], requested_metric: str):
    """
    Select best epoch using requested validation metric.

    average_precision and roc_auc are threshold-independent.
    If sklearn is unavailable or metric is None, fall back to balanced accuracy.
    """
    score = metrics.get(requested_metric)

    if score is None:
        return metrics["balanced_accuracy"], "balanced_accuracy"

    if isinstance(score, float) and math.isnan(score):
        return metrics["balanced_accuracy"], "balanced_accuracy"

    return float(score), requested_metric


def fmt_metric(value):
    if value is None:
        return "None"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline GNN using graph structure only. "
            "Includes Top-K ranking metrics and threshold adjustment analysis."
        )
    )

    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to graphs.pt.",
    )

    parser.add_argument(
        "--output",
        default="baseline_gnn_output",
        help="Output folder.",
    )

    parser.add_argument(
        "--gnn-type",
        default="GCN",
        choices=["GCN", "GIN", "GINE"],
        help="Baseline GNN type.",
    )

    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "sum", "max"],
        help="Graph pooling method.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

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
            "Default: average_precision, which is useful for imbalanced datasets."
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

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"GNN type: {args.gnn_type}")
    print(f"Pooling: {args.pooling}")
    print(f"Model-selection metric: {args.model_selection_metric}")
    print(f"Threshold-selection metric: {args.threshold_selection_metric}")
    print(f"Minimum recall for threshold selection: {args.min_recall_for_threshold}")

    graphs = load_graphs(args.graphs)

    node_dim, edge_dim = infer_dimensions_and_validate_graphs(graphs)

    inactive, active = count_labels(graphs)

    print(f"Number of graphs: {len(graphs)}")
    print(f"Inactive: {inactive}")
    print(f"Active: {active}")
    print(f"Overall active rate: {active / len(graphs):.6f}")
    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")

    if args.gnn_type == "GINE" and edge_dim is None:
        raise ValueError(
            "You selected GINE, but no edge_attr was found. "
            "Use GCN/GIN or regenerate graphs with edge features."
        )

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

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BaselineGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
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
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
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
                    "best_epoch": best_epoch,
                    "model_selection_metric_requested": args.model_selection_metric,
                    "model_selection_metric_used": best_metric_used,
                    "best_val_score": best_val_score,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
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
        model,
        test_loader,
        criterion,
        device,
        threshold=0.50,
    )

    print("\nApplying selected threshold to test set...")

    test_metrics_selected_threshold = evaluate(
        model,
        test_loader,
        criterion,
        device,
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
            "best_epoch": best_epoch,
            "model_selection_metric_requested": args.model_selection_metric,
            "model_selection_metric_used": best_metric_used,
            "best_val_score": best_val_score,
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
        "best_epoch": best_epoch,
        "model_selection_metric_requested": args.model_selection_metric,
        "model_selection_metric_used": best_metric_used,
        "best_val_score": best_val_score,
        "test_metrics_default_threshold_0.50": test_metrics_default,
        "selected_threshold": selected_threshold,
        "selected_threshold_info_path": selected_threshold_info_path,
        "test_metrics_selected_threshold": test_metrics_selected_threshold,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "history_path": history_path,
        "val_threshold_sweep_path": val_threshold_sweep_path,
        "selected_test_metrics_path": selected_test_metrics_path,
        "selected_test_confusion_matrix_path": selected_test_confusion_matrix_path,
        "test_output_paths": test_output_paths,
        "full_dataset_output_paths": full_dataset_output_paths,
    }

    with open(summary_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print("\nDone.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nImportant output files:")
    print(f"Test ranked predictions: {test_output_paths['ranked_predictions_path']}")
    print(f"Test Top-K metrics: {test_output_paths['topk_metrics_path']}")
    print(f"Test threshold sweep: {test_output_paths['threshold_metrics_path']}")
    print(f"Validation threshold sweep: {val_threshold_sweep_path}")
    print(f"Selected threshold info: {selected_threshold_info_path}")
    print(f"Selected-threshold test metrics: {selected_test_metrics_path}")
    print(f"Selected-threshold confusion matrix: {selected_test_confusion_matrix_path}")

    if full_dataset_output_paths is not None:
        print(f"Full-dataset ranked predictions: {full_dataset_output_paths['ranked_predictions_path']}")
        print(f"Full-dataset Top-K metrics: {full_dataset_output_paths['topk_metrics_path']}")


if __name__ == "__main__":
    main()