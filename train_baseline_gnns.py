#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import List, Tuple

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
    torch.cuda.manual_seed_all(seed)


def load_graphs(path: str):
    print(f"Loading graphs from: {path}")
    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError("Expected graphs.pt to contain a list of PyG Data objects.")

    cleaned = []
    skipped = 0

    for g in graphs:
        if not hasattr(g, "x") or not hasattr(g, "edge_index") or not hasattr(g, "y"):
            skipped += 1
            continue

        g.x = g.x.float()

        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            g.edge_attr = g.edge_attr.float()

        g.y = g.y.long().view(-1)

        cleaned.append(g)

    print(f"Loaded graphs: {len(cleaned)}")
    print(f"Skipped invalid graphs: {skipped}")

    return cleaned


def stratified_split(
    graphs: List,
    train_frac: float = 0.72,
    val_frac: float = 0.08,
    seed: int = 42,
) -> Tuple[List, List, List]:
    rng = random.Random(seed)

    inactive = [g for g in graphs if int(g.y.item()) == 0]
    active = [g for g in graphs if int(g.y.item()) == 1]

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
        edge_dim: int,
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
def evaluate(model, loader, criterion, device):
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
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())

    metrics = compute_basic_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / total_graphs

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(set(all_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(all_true, all_prob)
            metrics["average_precision"] = average_precision_score(all_true, all_prob)
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

    except Exception:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Collect predicted probabilities for each graph.

    This is needed for:
    - ranking compounds by predicted active probability
    - Precision@K
    - EF@K
    - threshold sweep
    """
    model.eval()

    rows = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds_default = torch.argmax(logits, dim=1)
        y_true = batch.y.view(-1)

        for i in range(batch.num_graphs):
            row = {
                "y_true": int(y_true[i].cpu().item()),
                "y_pred_default": int(preds_default[i].cpu().item()),
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
    EF@K = Precision@K / overall active rate
    """
    rows = []

    for k in k_values:
        if k > len(pred_df):
            print(f"Skipping K={k} because there are only {len(pred_df)} predictions.")
            continue

        top_k = pred_df.head(k)
        true_actives = int(top_k["y_true"].sum())

        precision_at_k = true_actives / k
        active_rate_top_k = precision_at_k

        ef_at_k = (
            precision_at_k / overall_active_rate
            if overall_active_rate > 0
            else None
        )

        rows.append(
            {
                "K": k,
                "true_actives_in_top_K": true_actives,
                "active_rate_in_top_K": active_rate_top_k,
                f"Precision@{k}": precision_at_k,
                "overall_active_rate": overall_active_rate,
                f"EF{k}": ef_at_k,
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(pred_df, thresholds=None):
    """
    Test different probability thresholds.

    Example:
    threshold = 0.50 means:
      prob_active >= 0.50 -> active
      prob_active < 0.50 -> inactive
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in torch.arange(0.05, 1.00, 0.05).tolist()]

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


def save_confusion_matrix(metrics, output_path):
    """
    Save confusion matrix in a readable layout.

    Rows = actual class
    Columns = predicted class
    """
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
):
    """
    Save ranked predictions, Precision@K, EF@K, threshold sweep, and confusion matrix.
    """
    pred_df = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    ranked_predictions_path = os.path.join(output_dir, f"{prefix}_ranked_predictions.csv")
    pred_df.to_csv(ranked_predictions_path, index=False)

    topk_df = calculate_topk_metrics(
        pred_df=pred_df,
        overall_active_rate=overall_active_rate,
        k_values=(100, 500, 1000),
    )

    topk_metrics_path = os.path.join(output_dir, f"{prefix}_topk_metrics.csv")
    topk_df.to_csv(topk_metrics_path, index=False)

    threshold_df = threshold_sweep(pred_df)

    threshold_metrics_path = os.path.join(output_dir, f"{prefix}_threshold_sweep_metrics.csv")
    threshold_df.to_csv(threshold_metrics_path, index=False)

    # Confusion matrix using the default model prediction, equivalent to argmax
    y_true = pred_df["y_true"].tolist()
    y_pred_default = pred_df["y_pred_default"].tolist()
    default_metrics = compute_basic_metrics(y_true, y_pred_default)

    confusion_matrix_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    confusion_matrix_df = save_confusion_matrix(default_metrics, confusion_matrix_path)

    print(f"\n{prefix} confusion matrix:")
    print(confusion_matrix_df)

    print(f"\n{prefix} Top-K metrics:")
    print(topk_df)

    print(f"\n{prefix} threshold sweep:")
    print(threshold_df)

    return {
        "ranked_predictions_path": ranked_predictions_path,
        "topk_metrics_path": topk_metrics_path,
        "threshold_metrics_path": threshold_metrics_path,
        "confusion_matrix_path": confusion_matrix_path,
    }


def count_labels(graphs):
    labels = torch.tensor([int(g.y.item()) for g in graphs])
    inactive = int((labels == 0).sum().item())
    active = int((labels == 1).sum().item())
    return inactive, active


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline GNN using graph structure only. No FPPool aggregation."
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss.",
    )

    parser.add_argument(
        "--save-full-dataset-ranking",
        action="store_true",
        help=(
            "Also rank all compounds in the full dataset. "
            "Use this if you want to satisfy the request to rank all compounds, "
            "but note that this includes training compounds."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"GNN type: {args.gnn_type}")
    print(f"Pooling: {args.pooling}")

    graphs = load_graphs(args.graphs)

    inactive, active = count_labels(graphs)

    print(f"Number of graphs: {len(graphs)}")
    print(f"Inactive: {inactive}")
    print(f"Active: {active}")

    first = graphs[0]

    node_dim = first.x.shape[1]

    if hasattr(first, "edge_attr") and first.edge_attr is not None and first.edge_attr.numel() > 0:
        edge_dim = first.edge_attr.shape[1]
    else:
        edge_dim = None

    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")

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
        class_weights = None
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

        print(f"Using class weights: inactive={weight_inactive:.4f}, active={weight_active:.4f}")

    best_val_metric = -1.0
    best_epoch = 0
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
        )

        val_score = val_metrics["balanced_accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)

        if val_score > best_val_metric:
            best_val_metric = val_score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "best_epoch": best_epoch,
                    "best_val_balanced_accuracy": best_val_metric,
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
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_auc={val_metrics['roc_auc']}"
        )

    print("\nLoading best model for final test evaluation...")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nSaving ranking-based test outputs...")

    test_active_rate = test_active / len(test_graphs)

    test_output_paths = save_ranking_outputs(
        model=model,
        loader=test_loader,
        device=device,
        output_dir=args.output,
        prefix="test",
        overall_active_rate=test_active_rate,
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
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "test_metrics": test_metrics,
        },
        final_model_path,
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

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
        "best_val_balanced_accuracy": best_val_metric,
        "test_metrics": test_metrics,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "history_path": history_path,
        "test_output_paths": test_output_paths,
        "full_dataset_output_paths": full_dataset_output_paths,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    print("\nSaved test output files:")
    print(json.dumps(test_output_paths, indent=2))

    if full_dataset_output_paths is not None:
        print("\nSaved full-dataset output files:")
        print(json.dumps(full_dataset_output_paths, indent=2))


if __name__ == "__main__":
    main()