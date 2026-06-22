#!/usr/bin/env python3

import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="Path to training_history.json")
    parser.add_argument("--output", default="overfitting_plots", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.history, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    print("\nColumns in history file:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    # Save readable CSV
    csv_path = os.path.join(args.output, "training_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    # Plot train loss and validation loss
    if "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.output, "loss_curve.png")
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")

    # Plot validation balanced accuracy
    if "val_balanced_accuracy" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_balanced_accuracy"], label="val_balanced_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced accuracy")
        plt.title("Validation Balanced Accuracy")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.output, "val_balanced_accuracy.png")
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")

        best_idx = df["val_balanced_accuracy"].idxmax()
        best_row = df.loc[best_idx]
        print("\nBest epoch based on validation balanced accuracy:")
        print(best_row)

    # Plot validation F1, recall, precision if available
    metric_cols = [
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
        "val_average_precision",
    ]

    available = [c for c in metric_cols if c in df.columns]

    if available:
        plt.figure()
        for c in available:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title("Validation Metrics")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.output, "validation_metrics.png")
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()