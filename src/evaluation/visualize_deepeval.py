#!/usr/bin/env python3
# visualize_deepeval.py
#
# Visualizes DeepEval result CSVs with columns:
# question,faithfulness_score,answer_relevancy_score,num_context_chunks
# and optionally: model_id
#
# If model_id is present, statistics are computed globally and per model.
#
# Global output (same as before):
# - scatter_faithfulness_vs_relevancy.png
# - hist_faithfulness_score.png
# - hist_answer_relevancy_score.png
# - lowest_faithfulness.png
# - summary.txt
# - below_threshold.csv  (rows with faithfulness < threshold, all models)
#
# Additional per-model output (if model_id column exists):
# - scatter_faithfulness_vs_relevancy_<MODEL>.png
# - hist_faithfulness_score_<MODEL>.png
# - hist_answer_relevancy_score_<MODEL>.png
# - lowest_faithfulness_<MODEL>.png
# - below_threshold_<MODEL>.csv
#
# where <MODEL> is a sanitized version of model_id.

import argparse
import os
import sys
from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def sanitize_model_id(model_id: str) -> str:
    """
    Sanitize model_id for use in filenames.
    Keeps alphanumerics, '-' and '_', replaces everything else with '_'.
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_id))


def read_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    base_expected = {"question", "faithfulness_score", "answer_relevancy_score", "num_context_chunks"}
    has_model = "model_id" in df.columns

    if has_model:
        expected = base_expected | {"model_id"}
    else:
        expected = base_expected

    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Ensure correct dtypes
    df["faithfulness_score"] = pd.to_numeric(df["faithfulness_score"], errors="coerce")
    df["answer_relevancy_score"] = pd.to_numeric(df["answer_relevancy_score"], errors="coerce")
    df["num_context_chunks"] = pd.to_numeric(df["num_context_chunks"], errors="coerce").astype("Int64")
    df["question"] = df["question"].astype(str)

    if has_model:
        df["model_id"] = df["model_id"].astype(str)
    else:
        # Backwards compatibility: treat as single model "ALL"
        df["model_id"] = "ALL"

    # Drop rows with NaNs in key columns
    df = df.dropna(subset=["faithfulness_score", "answer_relevancy_score", "num_context_chunks"])
    return df


def save_summary(df: pd.DataFrame, outdir: Path, faithfulness_threshold: float) -> None:
    # Global stats
    total = len(df)
    mean_f = df["faithfulness_score"].mean()
    mean_r = df["answer_relevancy_score"].mean()
    below = (df["faithfulness_score"] < faithfulness_threshold).sum()

    lines = [
        "DeepEval Summary",
        "----------------",
        f"Rows (all models): {total}",
        f"Mean faithfulness (all models): {mean_f:.3f}",
        f"Mean answer relevancy (all models): {mean_r:.3f}",
        f"Below faithfulness threshold ({faithfulness_threshold:.2f}) (all models): {below}",
        "",
        "Per-model statistics",
        "--------------------",
    ]

    for model_id, g in df.groupby("model_id"):
        m_total = len(g)
        m_mean_f = g["faithfulness_score"].mean()
        m_mean_r = g["answer_relevancy_score"].mean()
        m_below = (g["faithfulness_score"] < faithfulness_threshold).sum()
        lines.extend(
            [
                f"Model: {model_id}",
                f"  Rows: {m_total}",
                f"  Mean faithfulness: {m_mean_f:.3f}",
                f"  Mean answer relevancy: {m_mean_r:.3f}",
                f"  Below threshold ({faithfulness_threshold:.2f}): {m_below}",
                "",
            ]
        )

    summary = "\n".join(lines).rstrip()
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def plot_scatter(
    df: pd.DataFrame,
    outdir: Path,
    label: str | None = None,
    suffix: str = "",
) -> None:
    plt.figure()
    # Marker size scaled by context chunks (with a small multiplier for visibility)
    sizes = df["num_context_chunks"].fillna(0).astype(float) * 12 + 20
    plt.scatter(
        df["faithfulness_score"],
        df["answer_relevancy_score"],
        s=sizes,
        alpha=0.7,
        edgecolors="none",
    )
    plt.xlabel("Faithfulness score")
    plt.ylabel("Answer relevancy score")

    base_title = "Faithfulness vs. Answer Relevancy (marker size = num_context_chunks)"
    if label:
        plt.title(f"{base_title}\nModel: {label}")
    else:
        plt.title(base_title)

    plt.grid(True, linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = outdir / f"scatter_faithfulness_vs_relevancy{suffix}.png"
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_hist(
    df: pd.DataFrame,
    column: str,
    outdir: Path,
    label: str | None = None,
    suffix: str = "",
) -> None:
    plt.figure()
    plt.hist(df[column].dropna(), bins=20)
    pretty_name = column.replace("_", " ").capitalize()
    plt.xlabel(pretty_name)
    plt.ylabel("Count")

    base_title = f"Distribution of {pretty_name}"
    if label:
        plt.title(f"{base_title}\nModel: {label}")
    else:
        plt.title(base_title)

    plt.grid(True, linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    fname = outdir / f"hist_{column}{suffix}.png"
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_lowest_faithfulness(
    df: pd.DataFrame,
    outdir: Path,
    top_n: int,
    label: str | None = None,
    suffix: str = "",
) -> None:
    worst = df.nsmallest(top_n, "faithfulness_score")[["question", "faithfulness_score"]].copy()

    # Shorten long questions for labeling
    def shorten(q, limit=80):
        return q if len(q) <= limit else q[: limit - 1] + "…"

    labels = worst["question"].apply(shorten)
    scores = worst["faithfulness_score"]

    plt.figure(figsize=(10, max(3, 0.4 * len(worst) + 1)))
    y_pos = range(len(worst))
    plt.barh(list(y_pos), scores)
    plt.yticks(list(y_pos), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Faithfulness score")

    base_title = f"Lowest faithfulness questions (top {len(worst)})"
    if label:
        plt.title(f"{base_title}\nModel: {label}")
    else:
        plt.title(base_title)

    plt.grid(True, axis="x", linewidth=0.3)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    fname = outdir / f"lowest_faithfulness{suffix}.png"
    plt.savefig(fname, dpi=200)
    plt.close()


def write_below_threshold(
    df: pd.DataFrame,
    outdir: Path,
    threshold: float,
    suffix: str = "",
) -> None:
    bad = df[df["faithfulness_score"] < threshold].copy()
    bad.to_csv(outdir / f"below_threshold{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DeepEval results CSV (optionally per model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Path to DeepEval results CSV")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("deepeval_viz_out"),
        help="Output directory",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.9,
        help="Threshold to flag low faithfulness",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many lowest-faithfulness questions to show",
    )

    args = parser.parse_args()

    try:
        df = read_data(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Global summary & plots (all models together)
    save_summary(df, args.outdir, args.faithfulness_threshold)
    plot_scatter(df, args.outdir)  # global, no suffix
    plot_hist(df, "faithfulness_score", args.outdir)
    plot_hist(df, "answer_relevancy_score", args.outdir)
    plot_lowest_faithfulness(df, args.outdir, args.top_n)
    write_below_threshold(df, args.outdir, args.faithfulness_threshold)

    # Per-model plots & CSVs
    for model_id, g in df.groupby("model_id"):
        safe = sanitize_model_id(model_id)
        suffix = f"_{safe}"
        label = model_id

        plot_scatter(g, args.outdir, label=label, suffix=suffix)
        plot_hist(g, "faithfulness_score", args.outdir, label=label, suffix=suffix)
        plot_hist(g, "answer_relevancy_score", args.outdir, label=label, suffix=suffix)
        plot_lowest_faithfulness(g, args.outdir, args.top_n, label=label, suffix=suffix)
        write_below_threshold(g, args.outdir, args.faithfulness_threshold, suffix=suffix)

    print(f"\nWrote figures and files to: {args.outdir.resolve()}")
    print("Global files (all models combined):")
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

    print(
        "\nPer-model files have filenames suffixed with _<sanitized_model_id>, "
        "e.g. scatter_faithfulness_vs_relevancy_qwen_qwen3_235b_a22b_2507_v1_0.png"
    )


if __name__ == "__main__":
    main()
