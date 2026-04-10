"""Analyse the confidence gate using DeepEval CSV results.

Reads a CSV produced by ragas_eval.py (must contain gate_* columns) and
produces:

1. Abstention rate per experiment/config
2. Faithfulness comparison: confident vs. would-have-abstained (Mann-Whitney U)
3. Per-check diagnostic: which boolean checks predict answer quality
4. Threshold sensitivity sweep for top1_relevance and avg_top3_relevance

Usage:
    python -m src.evaluation.analyze_confidence_gate deepeval_results_short_public.csv \
        --outdir analysis_output/confidence_gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ── Column names used by ragas_eval.py ───────────────────────────────────────
GATE_NUMERIC_COLS = [
    "gate_num_chunks",
    "gate_num_unique_chunks",
    "gate_total_chars",
    "gate_top1_relevance",
    "gate_avg_top3_relevance",
]

METRIC_COLUMNS = [
    "faithfulness_score",
    "answer_relevancy_score",
    "contextual_relevancy_score",
    "contextual_recall_score",
    "contextual_precision_score",
]

# Default thresholds from ConfidenceChecker.__init__
DEFAULT_THRESHOLDS = {
    "gate_num_chunks": 2,
    "gate_num_unique_chunks": 1,
    "gate_total_chars": 200,
    "gate_top1_relevance": -4.0,
    "gate_avg_top3_relevance": -5.0,
}

CHECK_NAMES = {
    "gate_num_chunks": "enough_chunks",
    "gate_num_unique_chunks": "enough_unique_chunks",
    "gate_total_chars": "enough_text",
    "gate_top1_relevance": "top1_semantically_relevant",
    "gate_avg_top3_relevance": "top3_avg_semantically_relevant",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))


def read_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_gate = {"gate_confident"} | set(GATE_NUMERIC_COLS)
    missing = required_gate - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: CSV is missing gate columns: {missing}\n"
            "Re-run your evaluation with the latest ragas_eval.py to populate them."
        )

    # Coerce types
    for col in GATE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # gate_confident may be stored as string "True"/"False" or bool
    df["gate_confident"] = df["gate_confident"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    # Drop rows where gate data is entirely missing (e.g. resumed from old run)
    gate_present = df[GATE_NUMERIC_COLS].notna().any(axis=1)
    n_before = len(df)
    df = df[gate_present].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"NOTE: Dropped {n_dropped}/{n_before} rows with missing gate data (old run).")
    if df.empty:
        sys.exit("ERROR: No rows with populated gate data. Re-run evaluation first.")

    # Reconstruct individual boolean checks from numeric values
    for col, threshold in DEFAULT_THRESHOLDS.items():
        check_col = f"check_{CHECK_NAMES[col]}"
        df[check_col] = df[col] >= threshold

    # Fill missing experiment column
    if "experiment" not in df.columns:
        df["experiment"] = "ALL"

    return df


# ── 1. Abstention rate per config ────────────────────────────────────────────

def abstention_rate_table(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 70)
    print("1. ABSTENTION RATE PER EXPERIMENT")
    print("=" * 70)

    group_cols = ["experiment"]
    for optional in ("answer_prompt", "judge_prompt", "query_prompt"):
        if optional in df.columns:
            group_cols.append(optional)

    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for name, g in grouped:
        n = len(g)
        abstain_rate = (~g["gate_confident"].fillna(False).astype(bool)).mean()
        faith_mean = g["faithfulness_score"].mean()
        rows.append({
            **dict(zip(group_cols, name if isinstance(name, tuple) else [name])),
            "n": n,
            "abstention_rate": f"{abstain_rate:.2%}",
            "mean_faithfulness": f"{faith_mean:.3f}" if pd.notna(faith_mean) else "N/A",
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(outdir / "abstention_rate_per_config.csv", index=False)
    print(f"  -> Saved to {outdir / 'abstention_rate_per_config.csv'}")


# ── 2. Confident vs. abstained comparison ────────────────────────────────────

def confident_vs_abstained(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 70)
    print("2. CONFIDENT vs. WOULD-HAVE-ABSTAINED COMPARISON")
    print("=" * 70)

    metrics_present = [c for c in METRIC_COLUMNS if c in df.columns and df[c].notna().any()]

    confident = df[df["gate_confident"] == True]
    abstained = df[df["gate_confident"] == False]
    print(f"  Confident group:  n={len(confident)}")
    print(f"  Abstained group:  n={len(abstained)}")

    if len(confident) < 2 or len(abstained) < 2:
        print("  WARNING: Not enough samples in one group for statistical testing.")
        print("           Need at least 2 in each group.")
        return

    lines = []
    for metric in metrics_present:
        c_vals = confident[metric].dropna()
        a_vals = abstained[metric].dropna()
        if len(c_vals) < 2 or len(a_vals) < 2:
            continue

        c_mean = c_vals.mean()
        a_mean = a_vals.mean()
        delta = c_mean - a_mean

        stat, pval = mannwhitneyu(c_vals, a_vals, alternative="greater")

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        pretty = metric.replace("_score", "").replace("_", " ").title()
        line = (
            f"  {pretty:30s}  confident={c_mean:.3f}  abstained={a_mean:.3f}  "
            f"delta={delta:+.3f}  p={pval:.4f} {sig}"
        )
        print(line)
        lines.append({
            "metric": metric,
            "confident_mean": f"{c_mean:.3f}",
            "abstained_mean": f"{a_mean:.3f}",
            "delta": f"{delta:+.3f}",
            "p_value": f"{pval:.4f}",
            "significance": sig,
            "n_confident": len(c_vals),
            "n_abstained": len(a_vals),
        })

    if lines:
        pd.DataFrame(lines).to_csv(outdir / "confident_vs_abstained.csv", index=False)
        print(f"  -> Saved to {outdir / 'confident_vs_abstained.csv'}")

    # Box plot
    fig, axes = plt.subplots(1, len(metrics_present), figsize=(4 * len(metrics_present), 5))
    if len(metrics_present) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics_present):
        data = [
            confident[metric].dropna().values,
            abstained[metric].dropna().values,
        ]
        bp = ax.boxplot(data, labels=["Confident", "Abstained"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(metric.replace("_score", "").replace("_", " ").title(), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linewidth=0.3)
    fig.suptitle("Metric distributions: Confident vs. Would-have-abstained", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "confident_vs_abstained_boxplot.png", dpi=200)
    plt.close(fig)
    print(f"  -> Saved boxplot to {outdir / 'confident_vs_abstained_boxplot.png'}")


# ── 3. Per-check diagnostic ─────────────────────────────────────────────────

def per_check_diagnostic(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 70)
    print("3. PER-CHECK DIAGNOSTIC  (does each check predict faithfulness?)")
    print("=" * 70)

    faith = "faithfulness_score"
    if faith not in df.columns or df[faith].isna().all():
        print("  WARNING: No faithfulness scores available. Skipping.")
        return

    check_cols = [c for c in df.columns if c.startswith("check_")]
    rows = []

    for check in check_cols:
        passed = df[df[check] == True][faith].dropna()
        failed = df[df[check] == False][faith].dropna()

        if len(failed) == 0:
            print(f"  {check:40s}  NEVER FAILS — contributes nothing")
            rows.append({
                "check": check,
                "pass_mean": f"{passed.mean():.3f}",
                "fail_mean": "N/A",
                "n_pass": len(passed),
                "n_fail": 0,
                "delta": "N/A",
                "note": "never fails",
            })
            continue

        if len(passed) == 0:
            print(f"  {check:40s}  ALWAYS FAILS")
            rows.append({
                "check": check,
                "pass_mean": "N/A",
                "fail_mean": f"{failed.mean():.3f}",
                "n_pass": 0,
                "n_fail": len(failed),
                "delta": "N/A",
                "note": "always fails",
            })
            continue

        delta = passed.mean() - failed.mean()
        p_str = ""
        if len(passed) >= 2 and len(failed) >= 2:
            _, pval = mannwhitneyu(passed, failed, alternative="greater")
            p_str = f"  p={pval:.4f}"

        print(
            f"  {check:40s}  pass={passed.mean():.3f} (n={len(passed)})  "
            f"fail={failed.mean():.3f} (n={len(failed)})  delta={delta:+.3f}{p_str}"
        )
        rows.append({
            "check": check,
            "pass_mean": f"{passed.mean():.3f}",
            "fail_mean": f"{failed.mean():.3f}",
            "n_pass": len(passed),
            "n_fail": len(failed),
            "delta": f"{delta:+.3f}",
            "note": "",
        })

    if rows:
        pd.DataFrame(rows).to_csv(outdir / "per_check_diagnostic.csv", index=False)
        print(f"  -> Saved to {outdir / 'per_check_diagnostic.csv'}")


# ── 4. Threshold sensitivity ─────────────────────────────────────────────────

def threshold_sensitivity(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 70)
    print("4. THRESHOLD SENSITIVITY")
    print("=" * 70)

    faith = "faithfulness_score"
    if faith not in df.columns or df[faith].isna().all():
        print("  WARNING: No faithfulness scores available. Skipping.")
        return

    sweep_configs = [
        ("gate_top1_relevance", np.arange(-10.0, 2.0, 0.5)),
        ("gate_avg_top3_relevance", np.arange(-10.0, 2.0, 0.5)),
    ]

    for col, thresholds in sweep_configs:
        if col not in df.columns or df[col].isna().all():
            continue

        current_default = DEFAULT_THRESHOLDS[col]
        check_name = CHECK_NAMES[col]
        print(f"\n  Sweep: {check_name}  (column: {col}, default: {current_default})")

        rows = []
        accept_rates = []
        mean_faiths = []

        for t in thresholds:
            accepted = df[df[col] >= t]
            if len(accepted) == 0:
                continue
            rate = len(accepted) / len(df)
            mean_f = accepted[faith].mean()
            is_default = " <-- current" if abs(t - current_default) < 0.01 else ""
            print(f"    threshold >= {t:+6.1f}:  accept {rate:5.1%},  mean faithfulness={mean_f:.3f}{is_default}")
            rows.append({"threshold": t, "accept_rate": rate, "mean_faithfulness": mean_f})
            accept_rates.append(rate)
            mean_faiths.append(mean_f)

        if rows:
            sweep_df = pd.DataFrame(rows)
            sweep_df.to_csv(outdir / f"threshold_sweep_{_sanitize(check_name)}.csv", index=False)

            # Plot: dual-axis (accept rate + faithfulness vs. threshold)
            fig, ax1 = plt.subplots(figsize=(8, 5))
            color1 = "#1976D2"
            color2 = "#E64A19"

            ax1.set_xlabel(f"Threshold ({check_name})")
            ax1.set_ylabel("Accept rate", color=color1)
            ax1.plot(sweep_df["threshold"], sweep_df["accept_rate"], "o-", color=color1, label="Accept rate")
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.set_ylim(-0.05, 1.05)

            ax2 = ax1.twinx()
            ax2.set_ylabel("Mean faithfulness", color=color2)
            ax2.plot(sweep_df["threshold"], sweep_df["mean_faithfulness"], "s--", color=color2, label="Mean faithfulness")
            ax2.tick_params(axis="y", labelcolor=color2)
            ax2.set_ylim(-0.05, 1.05)

            # Mark default threshold
            ax1.axvline(current_default, color="gray", linestyle=":", linewidth=1.5, label=f"Default ({current_default})")
            ax1.legend(loc="lower left")
            ax2.legend(loc="lower right")

            fig.suptitle(f"Threshold sensitivity: {check_name}", fontsize=12)
            fig.tight_layout()
            fname = outdir / f"threshold_sweep_{_sanitize(check_name)}.png"
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f"  -> Saved plot to {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse confidence gate effectiveness from DeepEval CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Path to DeepEval results CSV with gate columns.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_output/confidence_gate"),
        help="Directory for output files (default: analysis_output/confidence_gate)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.csv)
    print(f"Loaded {len(df)} rows with gate data from {args.csv}")

    abstention_rate_table(df, args.outdir)
    confident_vs_abstained(df, args.outdir)
    per_check_diagnostic(df, args.outdir)
    threshold_sensitivity(df, args.outdir)

    print(f"\nAll outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
