#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GINEConv


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_repo(path: str):
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FPPool path not found: {path}\n"
            f"Example: --fppool-path ./FPPooling"
        )

    if path not in sys.path:
        sys.path.insert(0, path)


def get_pool_class(path: str):
    add_repo(path)

    try:
        from fppcode.pooling import FingerprintPool
    except Exception as error:
        raise ImportError(
            "Could not import FingerprintPool.\n"
            "Check that your FPPool repo path contains fppcode/pooling.py\n"
            f"Current path: {path}\n"
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    return FingerprintPool


def load_data(path: str):
    print(f"Loading graphs from: {path}")

    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError("Expected graphs.pt to contain a list of PyG Data objects.")

    data = []
    skipped = 0

    for g in graphs:
        missing = [name for name in ["x", "edge_index", "edge_attr", "y", "fp"] if not hasattr(g, name)]

        if missing:
            skipped += 1
            continue

        if g.fp.dim() != 2:
            skipped += 1
            continue

        g.x = g.x.float()
        g.edge_attr = g.edge_attr.float()
        g.y = g.y.long().view(-1)
        g.fp = g.fp.bool()

        data.append(g)

    print(f"Usable graphs: {len(data)}")
    print(f"Skipped graphs: {skipped}")

    if len(data) == 0:
        raise ValueError("No usable graphs found.")

    return data


def count_labels(data):
    y = torch.tensor([int(g.y.item()) for g in data])
    inactive = int((y == 0).sum().item())
    active = int((y == 1).sum().item())
    return inactive, active


def split_data(data, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)

    inactive = [g for g in data if int(g.y.item()) == 0]
    active = [g for g in data if int(g.y.item()) == 1]

    rng.shuffle(inactive)
    rng.shuffle(active)

    def split_one(items):
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]

        return train, val, test

    inactive_train, inactive_val, inactive_test = split_one(inactive)
    active_train, active_val, active_test = split_one(active)

    train = inactive_train + active_train
    val = inactive_val + active_val
    test = inactive_test + active_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def parse_lengths(text: Optional[str], fp_dim: int):
    if text is None:
        return None

    values = [int(x.strip()) for x in text.split(",") if x.strip()]

    if len(values) == 0:
        raise ValueError("--fp-lengths was provided but no values were found.")

    if sum(values) != fp_dim:
        raise ValueError(
            f"--fp-lengths must sum to the fingerprint dimension.\n"
            f"Given lengths: {values}\n"
            f"Sum: {sum(values)}\n"
            f"Fingerprint dimension: {fp_dim}"
        )

    return values


def get_fp_lengths(first_graph, fp_dim: int, text: Optional[str]):
    values = parse_lengths(text, fp_dim)

    if values is not None:
        print(f"Using fingerprint lengths from --fp-lengths: {values}")
        return values

    if hasattr(first_graph, "fp_length"):
        values = torch.as_tensor(first_graph.fp_length).view(-1).cpu().tolist()
        values = [int(v) for v in values if int(v) > 0]

        if len(values) > 0 and sum(values) == fp_dim:
            print(f"Using fingerprint lengths from graphs.pt: {values}")
            return values

    print("Could not read fingerprint groups from graphs.pt.")
    print("Using one fingerprint group containing all bits.")
    return [fp_dim]


class GNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int],
        hidden_dim: int,
        layers: int,
        dropout: float,
        gnn_type: str,
    ):
        super().__init__()

        self.gnn_type = gnn_type.upper()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.gnn_type == "GCN":
            self.convs.append(GCNConv(node_dim, hidden_dim))
            for _ in range(layers - 1):
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
            for _ in range(layers - 1):
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
                raise ValueError("GINE requires edge_attr.")

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
            for _ in range(layers - 1):
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
            raise ValueError("gnn-type must be GCN, GIN, or GINE.")

        for _ in range(layers):
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None):
        for conv, norm in zip(self.convs, self.norms):
            if self.gnn_type == "GINE":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class FPPoolModel(nn.Module):
    def __init__(
        self,
        pool_class,
        node_dim: int,
        edge_dim: Optional[int],
        hidden_dim: int,
        layers: int,
        dropout: float,
        gnn_type: str,
        fp_lengths: List[int],
        num_classes: int = 2,
    ):
        super().__init__()

        self.gnn = GNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            gnn_type=gnn_type,
        )

        self.pool = pool_class(
            in_channel=hidden_dim,
            hidden_channel=hidden_dim,
            fp_length=torch.tensor(fp_lengths, dtype=torch.long),
            reduce_inner="attn",
            reduce_inter="attn",
            reduce_global="attn",
            atoms_repr=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch):
        atom_h = self.gnn(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )

        graph_h, _, _ = self.pool(
            atom_h,
            batch.batch,
            batch.fp,
        )

        logits = self.classifier(graph_h)

        return logits


def get_metrics(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    accuracy = (y_true == y_pred).float().mean().item()

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
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


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        y = batch.y.view(-1)

        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs


@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_graphs = 0

    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        y = batch.y.view(-1)

        loss = loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    out = get_metrics(y_true, y_pred)
    out["loss"] = total_loss / total_graphs

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(set(y_true)) > 1:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
            out["average_precision"] = average_precision_score(y_true, y_prob)
        else:
            out["roc_auc"] = None
            out["average_precision"] = None

    except Exception:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train Model B: GNN + FPPool."
    )

    parser.add_argument("--graphs", required=True, help="Path to graphs.pt.")
    parser.add_argument("--fppool-path", default="./FPPooling", help="Path to the FPPool repo.")
    parser.add_argument("--output", default="fppool_output", help="Output folder.")

    parser.add_argument("--gnn-type", default="GINE", choices=["GCN", "GIN", "GINE"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-lengths", default=None, help="Example: 1024,1024,166")
    parser.add_argument("--no-class-weights", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: GNN + FPPool")
    print(f"GNN: {args.gnn_type}")

    Pool = get_pool_class(args.fppool_path)

    data = load_data(args.graphs)

    inactive, active = count_labels(data)

    first = data[0]
    node_dim = int(first.x.shape[1])
    edge_dim = int(first.edge_attr.shape[1]) if first.edge_attr is not None and first.edge_attr.numel() > 0 else None
    fp_dim = int(first.fp.shape[1])
    fp_lengths = get_fp_lengths(first, fp_dim, args.fp_lengths)

    print(f"Graphs: {len(data)}")
    print(f"Inactive: {inactive}")
    print(f"Active: {active}")
    print(f"Node dim: {node_dim}")
    print(f"Edge dim: {edge_dim}")
    print(f"FP dim: {fp_dim}")
    print(f"FP lengths: {fp_lengths}")

    train_data, val_data, test_data = split_data(
        data=data,
        train_frac=0.72,
        val_frac=0.08,
        seed=args.seed,
    )

    train_inactive, train_active = count_labels(train_data)
    val_inactive, val_active = count_labels(val_data)
    test_inactive, test_active = count_labels(test_data)

    print(f"Train: {len(train_data)} ({train_inactive}/{train_active})")
    print(f"Val: {len(val_data)} ({val_inactive}/{val_active})")
    print(f"Test: {len(test_data)} ({test_inactive}/{test_active})")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = FPPoolModel(
        pool_class=Pool,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        fp_lengths=fp_lengths,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.no_class_weights:
        loss_fn = nn.CrossEntropyLoss()
        print("Loss: CrossEntropyLoss")
    else:
        total = train_inactive + train_active
        weight_inactive = total / (2 * train_inactive)
        weight_active = total / (2 * train_active)

        weights = torch.tensor(
            [weight_inactive, weight_active],
            dtype=torch.float,
            device=device,
        )

        loss_fn = nn.CrossEntropyLoss(weight=weights)

        print(f"Loss: weighted CrossEntropyLoss")
        print(f"Weights: inactive={weight_inactive:.4f}, active={weight_active:.4f}")

    best_score = -1.0
    best_epoch = 0
    history = []

    best_path = os.path.join(args.output, "best.pt")
    final_path = os.path.join(args.output, "final.pt")
    history_path = os.path.join(args.output, "history.json")
    summary_path = os.path.join(args.output, "summary.json")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val = test(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        score = val["balanced_accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val.items()},
        }

        history.append(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "fp_dim": fp_dim,
                    "fp_lengths": fp_lengths,
                    "best_epoch": best_epoch,
                    "best_val_balanced_accuracy": best_score,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"val_acc={val['accuracy']:.4f} | "
            f"val_bal_acc={val['balanced_accuracy']:.4f} | "
            f"val_f1={val['f1']:.4f} | "
            f"val_recall={val['recall']:.4f} | "
            f"val_auc={val['roc_auc']}"
        )

    print("\nTesting best model...")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = test(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "fp_dim": fp_dim,
            "fp_lengths": fp_lengths,
            "test_metrics": test_metrics,
        },
        final_path,
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "model": "GNN + FPPool",
        "graphs": args.graphs,
        "gnn_type": args.gnn_type,
        "num_graphs": len(data),
        "inactive": inactive,
        "active": active,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "fp_dim": fp_dim,
        "fp_lengths": fp_lengths,
        "train_graphs": len(train_data),
        "val_graphs": len(val_data),
        "test_graphs": len(test_data),
        "train_inactive": train_inactive,
        "train_active": train_active,
        "val_inactive": val_inactive,
        "val_active": val_active,
        "test_inactive": test_inactive,
        "test_active": test_active,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_score,
        "test_metrics": test_metrics,
        "best_path": best_path,
        "final_path": final_path,
        "history_path": history_path,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Best model: {best_path}")
    print(f"Final model: {final_path}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
