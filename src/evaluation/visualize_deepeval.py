import argparse
import os
import sys
from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


# All known metric columns (some may or may not be present in a given CSV)
METRIC_COLUMNS = [
    "faithfulness_score",
    "answer_relevancy_score",
    "contextual_relevancy_score",
    "contextual_recall_score",
    "contextual_precision_score",
]


def sanitize_model_id(model_id: str) -> str:
    """
    Sanitize model_id (or any label) for use in filenames.
    Keeps alphanumerics, '-' and '_', replaces everything else with '_'.
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_id))


def read_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Minimal required columns for backwards compatibility
    base_expected = {"question", "faithfulness_score", "answer_relevancy_score", "num_context_chunks"}
    has_model = "model_id" in df.columns

    if has_model:
        expected = base_expected | {"model_id"}
    else:
        expected = base_expected

    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Ensure correct dtypes for all known metric columns that actually exist
    present_metrics = [col for col in METRIC_COLUMNS if col in df.columns]
    for col in present_metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["num_context_chunks"] = pd.to_numeric(df["num_context_chunks"], errors="coerce").astype("Int64")
    df["question"] = df["question"].astype(str)

    if has_model:
        df["model_id"] = df["model_id"].astype(str)
    else:
        # Backwards compatibility: treat as single model "ALL"
        df["model_id"] = "ALL"

    # New: optional experiment column
    if "experiment" in df.columns:
        df["experiment"] = df["experiment"].astype(str)
    else:
        df["experiment"] = "ALL"

    # questions_file is metadata only — keep it in the dataframe but never visualize it.
    if "questions_file" in df.columns:
        df["questions_file"] = df["questions_file"].astype(str)

    # Drop rows with NaNs in key columns (keep contextual metrics optional)
    df = df.dropna(subset=["faithfulness_score", "answer_relevancy_score", "num_context_chunks"])
    return df


def save_summary(df: pd.DataFrame, outdir: Path, faithfulness_threshold: float) -> None:
    metrics_present = [col for col in METRIC_COLUMNS if col in df.columns]

    # Global stats
    total = len(df)
    below = (df["faithfulness_score"] < faithfulness_threshold).sum()

    lines = [
        "DeepEval Summary",
        "----------------",
        f"Rows (all experiments, all models): {total}",
    ]

    # Global metric means
    for col in metrics_present:
        pretty = col.replace("_", " ").capitalize()
        mean_val = df[col].mean()
        lines.append(f"Mean {pretty} (global): {mean_val:.3f}")

    lines.append(f"Below faithfulness threshold ({faithfulness_threshold:.2f}) (global): {below}")
    lines.extend(
        [
            "",
            "Per-experiment statistics",
            "-------------------------",
        ]
    )

    # Per-experiment stats
    for experiment, gexp in df.groupby("experiment"):
        exp_total = len(gexp)
        exp_below = (gexp["faithfulness_score"] < faithfulness_threshold).sum()
        lines.append(f"Experiment: {experiment}")
        lines.append(f"  Rows: {exp_total}")
        for col in metrics_present:
            pretty = col.replace("_", " ").capitalize()
            mean_val = gexp[col].mean()
            lines.append(f"  Mean {pretty}: {mean_val:.3f}")
        lines.append(f"  Below threshold ({faithfulness_threshold:.2f}): {exp_below}")
        lines.append("")

    lines.extend(
        [
            "",
            "Per-model statistics (across all experiments)",
            "--------------------------------------------",
        ]
    )

    # Per-model stats (aggregating across experiments)
    for model_id, g in df.groupby("model_id"):
        m_total = len(g)
        m_below = (g["faithfulness_score"] < faithfulness_threshold).sum()
        lines.append(f"Model: {model_id}")
        lines.append(f"  Rows: {m_total}")

        for col in metrics_present:
            pretty = col.replace("_", " ").capitalize()
            mean_val = g[col].mean()
            lines.append(f"  Mean {pretty}: {mean_val:.3f}")

        lines.append(f"  Below threshold ({faithfulness_threshold:.2f}): {m_below}")
        lines.append("")

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
        plt.title(f"{base_title}\n{label}")
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
    if column not in df.columns:
        return

    plt.figure()
    plt.hist(df[column].dropna(), bins=20)
    pretty_name = column.replace("_", " ").capitalize()
    plt.xlabel(pretty_name)
    plt.ylabel("Count")

    base_title = f"Distribution of {pretty_name}"
    if label:
        plt.title(f"{base_title}\n{label}")
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
        plt.title(f"{base_title}\n{label}")
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

    # Exclude questions_file from the output "visualization" CSVs
    if "questions_file" in bad.columns:
        bad = bad.drop(columns=["questions_file"])

    bad.to_csv(outdir / f"below_threshold{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DeepEval results CSV (optionally per experiment and per model).",
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

    metrics_present = [col for col in METRIC_COLUMNS if col in df.columns]

    # Global summary & plots (all experiments, all models together)
    save_summary(df, args.outdir, args.faithfulness_threshold)
    plot_scatter(df, args.outdir)  # global, no suffix

    # Histograms for all present metrics (global)
    for col in metrics_present:
        plot_hist(df, col, args.outdir)

    plot_lowest_faithfulness(df, args.outdir, args.top_n)
    write_below_threshold(df, args.outdir, args.faithfulness_threshold)

    # Per-experiment plots & CSVs
    for experiment, gexp in df.groupby("experiment"):
        exp_safe = sanitize_model_id(experiment)
        exp_suffix = f"_{exp_safe}"
        exp_label = f"Experiment: {experiment}"

        plot_scatter(gexp, args.outdir, label=exp_label, suffix=exp_suffix)

        for col in metrics_present:
            plot_hist(gexp, col, args.outdir, label=exp_label, suffix=exp_suffix)

        plot_lowest_faithfulness(gexp, args.outdir, args.top_n, label=exp_label, suffix=exp_suffix)
        write_below_threshold(gexp, args.outdir, args.faithfulness_threshold, suffix=exp_suffix)

    # Per-(experiment, model) plots & CSVs
    for (experiment, model_id), g in df.groupby(["experiment", "model_id"]):
        exp_safe = sanitize_model_id(experiment)
        model_safe = sanitize_model_id(model_id)
        suffix = f"_{exp_safe}_{model_safe}"
        label = f"Model: {model_id} | Experiment: {experiment}"

        plot_scatter(g, args.outdir, label=label, suffix=suffix)

        # Histograms for all present metrics per (experiment, model)
        for col in metrics_present:
            plot_hist(g, col, args.outdir, label=label, suffix=suffix)

        plot_lowest_faithfulness(g, args.outdir, args.top_n, label=label, suffix=suffix)
        write_below_threshold(g, args.outdir, args.faithfulness_threshold, suffix=suffix)

    print(f"\nWrote figures and files to: {args.outdir.resolve()}")
    print("Global files (all experiments, all models combined):")

    # Always list the core files if they exist
    for name in [
        "summary.txt",
        "scatter_faithfulness_vs_relevancy.png",
        "lowest_faithfulness.png",
        "below_threshold.csv",
    ]:
        path = args.outdir / name
        if path.exists():
            print(f" - {name}")

    # Dynamically list all histogram files
    for hist_path in sorted(args.outdir.glob("hist_*.png")):
        print(f" - {hist_path.name}")

    print(
        "\nPer-experiment files have filenames suffixed with _<sanitized_experiment>, "
        "and per-(experiment, model) files with _<sanitized_experiment>_<sanitized_model_id>."
    )
    print(
        "Example: scatter_faithfulness_vs_relevancy_engineer_default_qwen_qwen3_235b_a22b_2507_v1_0.png"
    )


if __name__ == "__main__":
    main()
