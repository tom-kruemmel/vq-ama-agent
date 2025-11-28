#!/usr/bin/env python3
# visualize_deepeval.py
#
# Visualizes DeepEval result CSVs with columns:
# question,faithfulness_score,answer_relevancy_score,num_context_chunks
#
# Output:
# - scatter_faithfulness_vs_relevancy.png
# - hist_faithfulness.png
# - hist_relevancy.png
# - lowest_faithfulness.png
# - summary.txt
# - below_threshold.csv  (rows with faithfulness < threshold)

import argparse
import os
import sys
from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def read_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"question", "faithfulness_score", "answer_relevancy_score", "num_context_chunks"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Ensure correct dtypes
    df["faithfulness_score"] = pd.to_numeric(df["faithfulness_score"], errors="coerce")
    df["answer_relevancy_score"] = pd.to_numeric(df["answer_relevancy_score"], errors="coerce")
    df["num_context_chunks"] = pd.to_numeric(df["num_context_chunks"], errors="coerce").astype("Int64")
    df["question"] = df["question"].astype(str)

    # Drop rows with NaNs in key columns
    df = df.dropna(subset=["faithfulness_score", "answer_relevancy_score", "num_context_chunks"])
    return df


def save_summary(df: pd.DataFrame, outdir: Path, faithfulness_threshold: float) -> None:
    total = len(df)
    mean_f = df["faithfulness_score"].mean()
    mean_r = df["answer_relevancy_score"].mean()
    below = (df["faithfulness_score"] < faithfulness_threshold).sum()

    summary = textwrap.dedent(f"""
    DeepEval Summary
    ----------------
    Rows: {total}
    Mean faithfulness: {mean_f:.3f}
    Mean answer relevancy: {mean_r:.3f}
    Below faithfulness threshold ({faithfulness_threshold:.2f}): {below}
    """).strip()

    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def plot_scatter(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    # Marker size scaled by context chunks (with a small multiplier for visibility)
    sizes = df["num_context_chunks"].fillna(0).astype(float) * 12 + 20
    plt.scatter(df["faithfulness_score"], df["answer_relevancy_score"], s=sizes, alpha=0.7, edgecolors="none")
    plt.xlabel("Faithfulness score")
    plt.ylabel("Answer relevancy score")
    plt.title("Faithfulness vs. Answer Relevancy (marker size = num_context_chunks)")
    plt.grid(True, linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_faithfulness_vs_relevancy.png", dpi=200)
    plt.close()


def plot_hist(df: pd.DataFrame, column: str, outdir: Path) -> None:
    plt.figure()
    plt.hist(df[column].dropna(), bins=20)
    plt.xlabel(column.replace("_", " ").capitalize())
    plt.ylabel("Count")
    plt.title(f"Distribution of {column.replace('_', ' ')}")
    plt.grid(True, linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / f"hist_{column}.png", dpi=200)
    plt.close()


def plot_lowest_faithfulness(df: pd.DataFrame, outdir: Path, top_n: int) -> None:
    worst = df.nsmallest(top_n, "faithfulness_score")[["question", "faithfulness_score"]].copy()

    # Shorten long questions for labeling
    def shorten(q, limit=80):
        return (q if len(q) <= limit else q[:limit - 1] + "…")

    labels = worst["question"].apply(shorten)
    scores = worst["faithfulness_score"]

    plt.figure(figsize=(10, max(3, 0.4 * len(worst) + 1)))
    y_pos = range(len(worst))
    plt.barh(list(y_pos), scores)
    plt.yticks(list(y_pos), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Faithfulness score")
    plt.title(f"Lowest faithfulness questions (top {len(worst)})")
    plt.grid(True, axis="x", linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / "lowest_faithfulness.png", dpi=200)
    plt.close()


def write_below_threshold(df: pd.DataFrame, outdir: Path, threshold: float) -> None:
    bad = df[df["faithfulness_score"] < threshold].copy()
    bad.to_csv(outdir / "below_threshold.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DeepEval results CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Path to DeepEval results CSV")
    parser.add_argument("--outdir", type=Path, default=Path("deepeval_viz_out"), help="Output directory")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.9,
                        help="Threshold to flag low faithfulness")
    parser.add_argument("--top-n", type=int, default=10, help="How many lowest-faithfulness questions to show")

    args = parser.parse_args()

    try:
        df = read_data(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)

    save_summary(df, args.outdir, args.faithfulness_threshold)
    plot_scatter(df, args.outdir)
    plot_hist(df, "faithfulness_score", args.outdir)
    plot_hist(df, "answer_relevancy_score", args.outdir)
    plot_lowest_faithfulness(df, args.outdir, args.top_n)
    write_below_threshold(df, args.outdir, args.faithfulness_threshold)

    print(f"\nWrote figures and files to: {args.outdir.resolve()}")
    print("Files:")
    for name in [
        "summary.txt",
        "scatter_faithfulness_vs_relevancy.png",
        "hist_faithfulness_score.png",
        "hist_answer_relevancy_score.png",
        "lowest_faithfulness.png",
        "below_threshold.csv",
    ]:
        path = args.outdir / name
        if path.exists():
            print(f" - {name}")


if __name__ == "__main__":
    main()
