"""Evaluate the domain judge on a labelled test set.

Runs each question through DomainJudge with one or more prompt templates,
then reports precision / recall / F1 and plots score histograms.

Usage:
    poetry run python -m src.evaluation.evaluate_domain_judge \
        --test-set src/evaluation/domain_judge_test_set.csv \
        --outdir analysis_output/domain_judge

    # Single prompt only:
    poetry run python -m src.evaluation.evaluate_domain_judge \
        --test-set src/evaluation/domain_judge_test_set.csv \
        --prompts default

    # Specific model:
    poetry run python -m src.evaluation.evaluate_domain_judge \
        --test-set src/evaluation/domain_judge_test_set.csv \
        --model eu.amazon.nova-lite-v1:0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

load_dotenv()

from ..bedrock_client import BedrockClient
from ..domain_judge import DomainJudge
from ..prompt_templates import (
    JUDGE_DOMAIN_CATEGORY,
    JUDGE_DOMAIN_LENIENT,
    JUDGE_DOMAIN_TWO_STAGE,
    JUDGE_QUESTION_DOMAIN_PROMPT,
)

logger = logging.getLogger(__name__)

PROMPT_VARIANTS: dict[str, str] = {
    "default": JUDGE_QUESTION_DOMAIN_PROMPT,
    "lenient": JUDGE_DOMAIN_LENIENT,
    "two_stage": JUDGE_DOMAIN_TWO_STAGE,
    "category": JUDGE_DOMAIN_CATEGORY,
}


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))


# ── Load test set ────────────────────────────────────────────────────────────

def load_test_set(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row["expected_in_domain"].strip()
            rows.append({
                "question": row["question"].strip(),
                "expected_in_domain": expected.lower() == "true",
                "category": row.get("category", "").strip(),
            })
    return rows


# ── Run judge on test set ────────────────────────────────────────────────────

def run_judge(
    test_set: list[dict],
    judge: DomainJudge,
    min_score: float = 0.60,
    sleep: float = 0.5,
) -> pd.DataFrame:
    """Run the judge on every question and return a DataFrame of results."""
    records = []
    for i, item in enumerate(test_set):
        q = item["question"]
        expected = item["expected_in_domain"]
        category = item["category"]

        try:
            predicted, score, rationale = judge.judge(q, min_score=min_score)
        except Exception as exc:
            logger.warning("Judge call failed for %r: %s", q[:60], exc)
            predicted, score, rationale = False, 0.0, f"ERROR: {exc}"

        records.append({
            "question": q,
            "expected_in_domain": expected,
            "predicted_in_domain": predicted,
            "score": score,
            "rationale": rationale,
            "category": category,
        })
        logger.info(
            "[%2d/%d] %s  expected=%s predicted=%s score=%.2f",
            i + 1, len(test_set), q[:50], expected, predicted, score,
        )
        time.sleep(sleep)

    return pd.DataFrame(records)


# ── Classification report ────────────────────────────────────────────────────

def print_and_save_report(df: pd.DataFrame, prompt_name: str, outdir: Path) -> None:
    y_true = df["expected_in_domain"].astype(int)
    y_pred = df["predicted_in_domain"].astype(int)

    report = classification_report(
        y_true, y_pred,
        target_names=["out-of-domain", "in-domain"],
        digits=3,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"Prompt variant: {prompt_name}")
    print("=" * 60)
    print(report)
    print(f"Confusion matrix (rows=actual, cols=predicted):\n{cm}\n")

    # Save detailed results
    df.to_csv(outdir / f"results_{_sanitize(prompt_name)}.csv", index=False)

    # Save report text
    report_path = outdir / f"report_{_sanitize(prompt_name)}.txt"
    with open(report_path, "w") as f:
        f.write(f"Prompt variant: {prompt_name}\n\n")
        f.write(report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted):\n{cm}\n")
    print(f"  -> Saved report to {report_path}")


# ── Score histogram ──────────────────────────────────────────────────────────

def plot_score_histogram(df: pd.DataFrame, prompt_name: str, outdir: Path) -> None:
    in_domain = df[df["expected_in_domain"] == True]["score"]
    out_domain = df[df["expected_in_domain"] == False]["score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)

    ax.hist(in_domain, bins=bins, alpha=0.7, label="In-domain (expected)", color="#4CAF50", edgecolor="white")
    ax.hist(out_domain, bins=bins, alpha=0.7, label="Out-of-domain (expected)", color="#F44336", edgecolor="white")

    ax.axvline(0.60, color="black", linestyle="--", linewidth=1.5, label="Threshold (0.60)")
    ax.set_xlabel("Domain judge score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score distribution by class — {prompt_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", linewidth=0.3)

    fig.tight_layout()
    fname = outdir / f"score_histogram_{_sanitize(prompt_name)}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  -> Saved histogram to {fname}")


# ── Threshold sweep ──────────────────────────────────────────────────────────

def plot_threshold_f1(df: pd.DataFrame, prompt_name: str, outdir: Path) -> None:
    """Sweep min_score thresholds and plot F1 to find the optimal cutoff."""
    y_true = df["expected_in_domain"].astype(int).values
    scores = df["score"].values

    thresholds = np.arange(0.0, 1.01, 0.05)
    f1s = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, "o-", color="#1976D2", markersize=4)
    ax.axvline(0.60, color="gray", linestyle="--", linewidth=1, label="Current (0.60)")
    ax.axvline(best_t, color="#E64A19", linestyle=":", linewidth=1.5, label=f"Best ({best_t:.2f}, F1={best_f1:.3f})")
    ax.set_xlabel("min_score threshold")
    ax.set_ylabel("F1 score (in-domain class)")
    ax.set_title(f"Threshold vs. F1 — {prompt_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linewidth=0.3)

    fig.tight_layout()
    fname = outdir / f"threshold_f1_{_sanitize(prompt_name)}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  -> Saved threshold sweep to {fname}  (best threshold={best_t:.2f}, F1={best_f1:.3f})")


# ── Cross-prompt comparison ──────────────────────────────────────────────────

def plot_comparison(all_results: dict[str, pd.DataFrame], outdir: Path) -> None:
    """Bar chart comparing F1 across prompt variants."""
    if len(all_results) < 2:
        return

    names = []
    f1_in = []
    f1_out = []
    for name, df in all_results.items():
        y_true = df["expected_in_domain"].astype(int)
        y_pred = df["predicted_in_domain"].astype(int)
        names.append(name)
        f1_in.append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1_out.append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.5), 5))
    ax.bar(x - width / 2, f1_in, width, label="F1 (in-domain)", color="#4CAF50")
    ax.bar(x + width / 2, f1_out, width, label="F1 (out-of-domain)", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Domain judge prompt comparison")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linewidth=0.3)

    fig.tight_layout()
    fname = outdir / "prompt_comparison_f1.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"\n  -> Saved prompt comparison to {fname}")

    # Also save a summary CSV
    summary = pd.DataFrame({"prompt": names, "f1_in_domain": f1_in, "f1_out_of_domain": f1_out})
    summary.to_csv(outdir / "prompt_comparison_summary.csv", index=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate domain judge precision/recall/F1 on a labelled test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-set", type=Path,
        default=Path("src/evaluation/domain_judge_test_set.csv"),
        help="Path to labelled domain judge test set CSV.",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("analysis_output/domain_judge"),
        help="Output directory (default: analysis_output/domain_judge).",
    )
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="Prompt variants to evaluate (default: all). "
             f"Choose from: {', '.join(PROMPT_VARIANTS)}",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Bedrock model ID for the judge (default: auto-pick from env).",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.60,
        help="min_score threshold passed to judge (default: 0.60).",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="Seconds to sleep between API calls (default: 0.5).",
    )
    args = parser.parse_args()

    if not args.test_set.is_file():
        sys.exit(f"ERROR: Test set not found: {args.test_set}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Determine which prompts to run
    prompt_names = args.prompts or list(PROMPT_VARIANTS.keys())
    invalid = set(prompt_names) - set(PROMPT_VARIANTS)
    if invalid:
        sys.exit(f"ERROR: Unknown prompt variants: {invalid}. Choose from: {list(PROMPT_VARIANTS)}")

    # Load test set
    test_set = load_test_set(args.test_set)
    n_in = sum(1 for t in test_set if t["expected_in_domain"])
    n_out = len(test_set) - n_in
    print(f"Loaded {len(test_set)} questions ({n_in} in-domain, {n_out} out-of-domain)")

    # Init Bedrock client
    bedrock = BedrockClient()
    model_id = args.model or os.getenv("JUDGE_MODEL_ID", "qwen.qwen3-235b-a22b-2507-v1:0")
    print(f"Using model: {model_id}")

    # Run each prompt variant
    all_results: dict[str, pd.DataFrame] = {}

    for prompt_name in prompt_names:
        print(f"\n{'─' * 60}")
        print(f"Running prompt variant: {prompt_name}")
        print(f"{'─' * 60}")

        judge = DomainJudge(
            bedrock_client=bedrock,
            model_id=model_id,
            prompt_template=PROMPT_VARIANTS[prompt_name],
        )

        df = run_judge(test_set, judge, min_score=args.min_score, sleep=args.sleep)
        all_results[prompt_name] = df

        print_and_save_report(df, prompt_name, args.outdir)
        plot_score_histogram(df, prompt_name, args.outdir)
        plot_threshold_f1(df, prompt_name, args.outdir)

    # Cross-prompt comparison
    plot_comparison(all_results, args.outdir)

    print(f"\nAll outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
