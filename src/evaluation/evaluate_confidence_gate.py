"""Evaluate the confidence gate on constructed test cases.

Builds test cases automatically from the question CSV files: questions whose
expected answer is the default abstention message (in the public-only setting)
should be rejected by the confidence gate, while questions with real answers
should pass.

Sweeps all combinations of ConfidenceChecker parameters and reports
precision / recall / F1 and optimal thresholds.

Usage:
    # Run with defaults (all question files, default query prompt):
    poetry run python -m src.evaluation.evaluate_confidence_gate \
        --outdir analysis_output/confidence_gate_eval

    # Specific query prompt:
    poetry run python -m src.evaluation.evaluate_confidence_gate \
        --query-prompt diverse

    # Custom parameter grid:
    poetry run python -m src.evaluation.evaluate_confidence_gate \
        --min-top1 -6.0 -4.0 -2.0 0.0 \
        --min-avg-top3 -6.0 -4.0 -2.0 \
        --min-chunks 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, f1_score

load_dotenv()

from ..bedrock_client import BedrockClient
from ..embeddings import Embeddings
from ..rank_fusion import RankFusion
from ..reranker import CrossEncoderReranker, RerankedChunk
from ..confidence_checker import ConfidenceChecker
from ..agent import RAGAgent
from ..prompt_templates import (
    GENERATE_QUERIES_PROMPT,
    GENERATE_QUERIES_DIVERSE,
    GENERATE_QUERIES_HYDE,
    GENERATE_QUERIES_DECOMPOSE,
    GENERATE_ANSWER_PROMPT,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS: list[str] = [
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "eu.amazon.nova-pro-v1:0",
    "openai.gpt-oss-120b-1:0",
    "eu.amazon.nova-lite-v1:0",
    "eu.mistral.pixtral-large-2502-v1:0",
]

QUERY_PROMPT_VARIANTS = {
    "default": GENERATE_QUERIES_PROMPT,
    "diverse": GENERATE_QUERIES_DIVERSE,
    "hyde": GENERATE_QUERIES_HYDE,
    "decompose": GENERATE_QUERIES_DECOMPOSE,
}

DEFAULT_TEST_SET = "src/evaluation/confidence_gate_test_set.csv"


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))


# ── Load test set ────────────────────────────────────────────────────────────

def load_test_cases(test_set_path: str) -> list[dict]:
    """Load the confidence gate test set CSV.

    Expected columns: question, expected_confident, lang, headings, category
    """
    rows = []
    with open(test_set_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "expected_confident": row["expected_confident"].strip().lower() == "true",
                "lang": row.get("lang", "en").strip(),
                "headings": [h.strip() for h in row.get("headings", "PUBLIC").split(",")],
                "category": row.get("category", "").strip(),
            })
    return rows


# ── Retrieve and rerank (shared across threshold sweeps) ─────────────────────

def retrieve_and_rerank(
    test_cases: list[dict],
    *,
    agent: RAGAgent,
    retriever: Embeddings,
    fusion: RankFusion,
    reranker: CrossEncoderReranker,
    headings: list[str],
    top_k: int,
    rerank_top_n: int,
    sleep: float = 0.5,
) -> list[dict]:
    """Run retrieval + reranking for each test case and store the reranked chunks.

    Returns the test_cases list augmented with 'reranked_chunks' and
    'confidence_details_raw' (scores only, no threshold applied yet).
    """
    for i, tc in enumerate(test_cases):
        q = tc["question"]

        # Query expansion
        queries = agent.generate_queries(q)

        # Retrieval
        retrieved_docs = retriever.retrieve_documents(queries, headings, top_k=top_k)

        # RRF
        fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)

        # Cross-encoder reranking
        reranked = reranker.rerank(q, fused_docs, top_n=rerank_top_n)

        # Store raw features (independent of thresholds)
        if reranked:
            texts = [d.text for d in reranked]
            ce_scores = [d.ce_score for d in reranked]
            top3_scores = ce_scores[:3]
            tc["reranked_chunks"] = reranked
            tc["num_chunks"] = len(reranked)
            tc["num_unique_chunks"] = len(set(texts))
            tc["total_chars"] = sum(len(t) for t in texts)
            tc["top1_relevance"] = ce_scores[0]
            tc["avg_top3_relevance"] = sum(top3_scores) / len(top3_scores)
        else:
            tc["reranked_chunks"] = []
            tc["num_chunks"] = 0
            tc["num_unique_chunks"] = 0
            tc["total_chars"] = 0
            tc["top1_relevance"] = 0.0
            tc["avg_top3_relevance"] = 0.0

        logger.info(
            "[%2d/%d] %s  expected_confident=%s  top1=%.3f  avg_top3=%.3f  chunks=%d",
            i + 1, len(test_cases), q[:50],
            tc["expected_confident"],
            tc["top1_relevance"],
            tc["avg_top3_relevance"],
            tc["num_chunks"],
        )
        time.sleep(sleep)

    return test_cases


# ── Threshold sweep ──────────────────────────────────────────────────────────

def sweep_thresholds(
    test_cases: list[dict],
    *,
    min_chunks_values: list[int],
    min_unique_chunks_values: list[int],
    min_total_chars_values: list[int],
    min_top1_values: list[float],
    min_avg_top3_values: list[float],
) -> pd.DataFrame:
    """Sweep all parameter combos over pre-computed retrieval features."""
    records = []
    combos = list(itertools.product(
        min_chunks_values,
        min_unique_chunks_values,
        min_total_chars_values,
        min_top1_values,
        min_avg_top3_values,
    ))
    logger.info("Sweeping %d parameter combinations over %d test cases",
                len(combos), len(test_cases))

    for min_ch, min_uq, min_tc, min_t1, min_avg3 in combos:
        checker = ConfidenceChecker(
            min_chunks=min_ch,
            min_unique_chunks=min_uq,
            min_total_chars=min_tc,
            min_top1_relevance=min_t1,
            min_avg_top3_relevance=min_avg3,
        )

        y_true = []
        y_pred = []
        for tc in test_cases:
            confident, _ = checker.evaluate(tc["reranked_chunks"])
            y_true.append(tc["expected_confident"])
            y_pred.append(confident)

        y_true_int = [int(v) for v in y_true]
        y_pred_int = [int(v) for v in y_pred]

        # Compute metrics
        # "confident" is the positive class (should pass gate)
        f1_pass = f1_score(y_true_int, y_pred_int, pos_label=1, zero_division=0)
        # "not confident" = should abstain
        f1_abstain = f1_score(y_true_int, y_pred_int, pos_label=0, zero_division=0)
        f1_macro = f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)

        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

        n_total = len(test_cases)
        n_abstained = sum(1 for p in y_pred if not p)
        abstention_rate = n_abstained / n_total if n_total else 0

        records.append({
            "min_chunks": min_ch,
            "min_unique_chunks": min_uq,
            "min_total_chars": min_tc,
            "min_top1_relevance": min_t1,
            "min_avg_top3_relevance": min_avg3,
            "f1_pass": round(f1_pass, 4),
            "f1_abstain": round(f1_abstain, 4),
            "f1_macro": round(f1_macro, 4),
            "accuracy": round((tp + tn) / n_total, 4) if n_total else 0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "abstention_rate": round(abstention_rate, 4),
            "n_total": n_total,
        })

    return pd.DataFrame(records)


# ── Reporting ────────────────────────────────────────────────────────────────

def report_best(df: pd.DataFrame, outdir: Path) -> None:
    """Print and save the best configs by f1_macro."""
    df_sorted = df.sort_values("f1_macro", ascending=False)

    print(f"\n{'=' * 80}")
    print("Top 10 configurations by F1 (macro)")
    print("=" * 80)
    print(df_sorted.head(10).to_string(index=False))

    df_sorted.to_csv(outdir / "sweep_results.csv", index=False)
    print(f"\n  -> Full sweep results saved to {outdir / 'sweep_results.csv'}")

    # Best config detail
    best = df_sorted.iloc[0]
    print(f"\n{'─' * 60}")
    print("Best configuration:")
    print(f"  min_chunks           = {int(best['min_chunks'])}")
    print(f"  min_unique_chunks    = {int(best['min_unique_chunks'])}")
    print(f"  min_total_chars      = {int(best['min_total_chars'])}")
    print(f"  min_top1_relevance   = {best['min_top1_relevance']}")
    print(f"  min_avg_top3_relevance = {best['min_avg_top3_relevance']}")
    print(f"  F1 (macro)           = {best['f1_macro']}")
    print(f"  F1 (pass)            = {best['f1_pass']}")
    print(f"  F1 (abstain)         = {best['f1_abstain']}")
    print(f"  Accuracy             = {best['accuracy']}")
    print(f"  Abstention rate      = {best['abstention_rate']}")
    print(f"  TP={int(best['tp'])} TN={int(best['tn'])} FP={int(best['fp'])} FN={int(best['fn'])}")
    print(f"{'─' * 60}")


def save_per_question_details(test_cases: list[dict], outdir: Path, filename: str = "per_question_features.csv") -> None:
    """Save per-question retrieval features for analysis."""
    records = []
    for tc in test_cases:
        records.append({
            "question": tc["question"],
            "lang": tc["lang"],
            "category": tc.get("category", ""),
            "headings": ",".join(tc.get("headings", [])),
            "expected_confident": tc["expected_confident"],
            "num_chunks": tc["num_chunks"],
            "num_unique_chunks": tc["num_unique_chunks"],
            "total_chars": tc["total_chars"],
            "top1_relevance": round(tc["top1_relevance"], 4),
            "avg_top3_relevance": round(tc["avg_top3_relevance"], 4),
        })
    df = pd.DataFrame(records)
    df.to_csv(outdir / filename, index=False)
    print(f"  -> Per-question features saved to {outdir / filename}")


def plot_score_distributions(test_cases: list[dict], outdir: Path) -> None:
    """Histogram of top1_relevance and avg_top3_relevance by expected class."""
    for metric_key, label in [
        ("top1_relevance", "Top-1 CE score"),
        ("avg_top3_relevance", "Avg top-3 CE score"),
    ]:
        should_pass = [tc[metric_key] for tc in test_cases if tc["expected_confident"]]
        should_abstain = [tc[metric_key] for tc in test_cases if not tc["expected_confident"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = should_pass + should_abstain
        if not all_vals:
            plt.close(fig)
            continue
        lo = min(all_vals) - 0.5
        hi = max(all_vals) + 0.5
        bins = np.linspace(lo, hi, 25)

        ax.hist(should_pass, bins=bins, alpha=0.7, label="Should pass (real answer)", color="#4CAF50", edgecolor="white")
        ax.hist(should_abstain, bins=bins, alpha=0.7, label="Should abstain", color="#F44336", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label} by expected gate decision")
        ax.legend()
        ax.grid(axis="y", linewidth=0.3)
        fig.tight_layout()

        fname = outdir / f"distribution_{_sanitize(metric_key)}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"  -> Saved distribution plot to {fname}")


def plot_threshold_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Heatmap of f1_macro over min_top1_relevance × min_avg_top3_relevance
    (averaged over other params)."""
    if df.empty:
        return

    pivot = df.groupby(["min_top1_relevance", "min_avg_top3_relevance"])["f1_macro"].mean().unstack()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("min_avg_top3_relevance")
    ax.set_ylabel("min_top1_relevance")
    ax.set_title("F1 (macro) by relevance thresholds")
    fig.colorbar(im, ax=ax, label="F1 macro")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if pivot.values[i, j] > 0.5 else "white")

    fig.tight_layout()
    fname = outdir / "heatmap_relevance_thresholds.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  -> Saved heatmap to {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate confidence gate parameters on constructed test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-set", type=Path,
        default=Path(DEFAULT_TEST_SET),
        help=f"Path to labelled confidence gate test set CSV (default: {DEFAULT_TEST_SET}).",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("analysis_output/confidence_gate_eval"),
        help="Output directory (default: analysis_output/confidence_gate_eval).",
    )
    parser.add_argument(
        "--query-prompts", nargs="*", default=None,
        choices=list(QUERY_PROMPT_VARIANTS),
        help="Query expansion prompts to test (default: all). "
             f"Choose from: {', '.join(QUERY_PROMPT_VARIANTS)}",
    )
    parser.add_argument(
        "--top-k", type=int, default=15,
        help="Number of documents to retrieve per query (default: 15).",
    )
    parser.add_argument(
        "--rerank-top-n", type=int, default=10,
        help="Number of documents to keep after reranking (default: 10).",
    )
    parser.add_argument(
        "--models", nargs="*", type=str, default=None,
        help="Bedrock model IDs for query expansion "
             f"(default: {', '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="Seconds to sleep between API calls (default: 0.5).",
    )
    # Confidence gate parameter grids
    parser.add_argument(
        "--min-chunks", nargs="+", type=int, default=[1, 2, 3],
        help="Values to sweep for min_chunks (default: 1 2 3).",
    )
    parser.add_argument(
        "--min-unique-chunks", nargs="+", type=int, default=[1],
        help="Values to sweep for min_unique_chunks (default: 1).",
    )
    parser.add_argument(
        "--min-total-chars", nargs="+", type=int, default=[100, 200, 400],
        help="Values to sweep for min_total_chars (default: 100 200 400).",
    )
    parser.add_argument(
        "--min-top1", nargs="+", type=float,
        default=[-6.0, -4.0, -2.0, 0.0, 2.0],
        help="Values to sweep for min_top1_relevance (default: -6 -4 -2 0 2).",
    )
    parser.add_argument(
        "--min-avg-top3", nargs="+", type=float,
        default=[-6.0, -5.0, -4.0, -3.0, -2.0],
        help="Values to sweep for min_avg_top3_relevance (default: -6 -5 -4 -3 -2).",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.test_set.is_file():
        sys.exit(f"ERROR: Test set not found: {args.test_set}")

    # Load test cases
    all_test_cases = load_test_cases(str(args.test_set))

    n_pass = sum(1 for tc in all_test_cases if tc["expected_confident"])
    n_abstain = len(all_test_cases) - n_pass
    print(f"Loaded {len(all_test_cases)} test cases "
          f"({n_pass} should pass, {n_abstain} should abstain)")

    if not all_test_cases:
        sys.exit("ERROR: No test cases loaded.")

    # Initialize shared components (retriever, fusion, reranker are prompt-agnostic)
    bedrock = BedrockClient()
    model_ids = args.models or DEFAULT_MODELS
    retriever = Embeddings()
    fusion = RankFusion()
    reranker = CrossEncoderReranker()

    # Determine which query prompts to test
    query_prompt_names = args.query_prompts or list(QUERY_PROMPT_VARIANTS)
    print(f"Query prompts to test: {query_prompt_names}")
    print(f"Models to test: {model_ids}")
    print(f"top_k={args.top_k}, rerank_top_n={args.rerank_top_n}")

    from collections import defaultdict
    import copy

    for model_id in model_ids:
        # Short model label for directory names
        model_label = model_id.split(".", 1)[-1].split(":")[0]  # e.g. "qwen3-235b-a22b-2507-v1"
        model_outdir = args.outdir / _sanitize(model_label)
        model_outdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'╔' + '═' * 68 + '╗'}")
        print(f"  Model: {model_id}")
        print(f"{'╚' + '═' * 68 + '╝'}")

        # Track best config per query prompt for cross-prompt comparison
        cross_prompt_results: dict[str, dict] = {}

        for qp_name in query_prompt_names:
            print(f"\n{'═' * 60}")
            print(f"Query prompt: {qp_name}  (model: {model_label})")
            print(f"{'═' * 60}")

            qp_outdir = model_outdir / _sanitize(qp_name)
            qp_outdir.mkdir(parents=True, exist_ok=True)

            query_prompt = QUERY_PROMPT_VARIANTS[qp_name]
            agent = RAGAgent(
                bedrock,
                model_id,
                generate_queries_prompt=query_prompt,
                generate_answer_prompt=GENERATE_ANSWER_PROMPT,
            )

            # Deep copy test cases so each prompt gets fresh feature slots
            qp_test_cases = copy.deepcopy(all_test_cases)

            # Group test cases by headings to batch retrieval
            by_headings = defaultdict(list)
            for tc in qp_test_cases:
                key = tuple(tc["headings"])
                by_headings[key].append(tc)

            print(f"\nRetrieving and reranking for {len(qp_test_cases)} questions...")
            for headings_tuple, cases in by_headings.items():
                headings = list(headings_tuple)
                print(f"\n  Headings: {headings} ({len(cases)} questions)")
                retrieve_and_rerank(
                    cases,
                    agent=agent,
                    retriever=retriever,
                    fusion=fusion,
                    reranker=reranker,
                    headings=headings,
                    top_k=args.top_k,
                    rerank_top_n=args.rerank_top_n,
                    sleep=args.sleep,
                )

            # ── Bidirectional check: confidential_only with CONFIDENTIAL ──
            confidential_only_cases = [
                tc for tc in qp_test_cases if tc["category"] == "confidential_only"
            ]
            confidential_mirror_cases = []
            if confidential_only_cases:
                print(f"\n  Running bidirectional check: {len(confidential_only_cases)} "
                      f"confidential_only questions with PUBLIC+CONFIDENTIAL headings...")
                for tc in confidential_only_cases:
                    mirror = {
                        "question": tc["question"],
                        "expected_confident": True,
                        "lang": tc["lang"],
                        "headings": ["PUBLIC", "CONFIDENTIAL"],
                        "category": "confidential_mirror",
                    }
                    confidential_mirror_cases.append(mirror)

                retrieve_and_rerank(
                    confidential_mirror_cases,
                    agent=agent,
                    retriever=retriever,
                    fusion=fusion,
                    reranker=reranker,
                    headings=["PUBLIC", "CONFIDENTIAL"],
                    top_k=args.top_k,
                    rerank_top_n=args.rerank_top_n,
                    sleep=args.sleep,
                )

            # Combine for sweep
            combined = qp_test_cases + confidential_mirror_cases
            print(f"\nTotal test cases for sweep: {len(combined)} "
                  f"({len(qp_test_cases)} original + {len(confidential_mirror_cases)} mirror)")

            # Save per-question features
            save_per_question_details(qp_test_cases, qp_outdir)
            if confidential_mirror_cases:
                save_per_question_details(
                    confidential_mirror_cases, qp_outdir,
                    filename="per_question_features_confidential_mirror.csv",
                )

            # Plot score distributions
            plot_score_distributions(qp_test_cases, qp_outdir)

            # Sweep thresholds
            n_combos = (len(args.min_chunks) * len(args.min_unique_chunks) *
                        len(args.min_total_chars) * len(args.min_top1) * len(args.min_avg_top3))
            print(f"\nSweeping {n_combos} parameter combinations...")

            sweep_df = sweep_thresholds(
                combined,
                min_chunks_values=args.min_chunks,
                min_unique_chunks_values=args.min_unique_chunks,
                min_total_chars_values=args.min_total_chars,
                min_top1_values=args.min_top1,
                min_avg_top3_values=args.min_avg_top3,
            )

            # Report best configurations
            report_best(sweep_df, qp_outdir)

            # Plot heatmap
            plot_threshold_heatmap(sweep_df, qp_outdir)

            # Track best config for cross-prompt comparison
            best = sweep_df.sort_values("f1_macro", ascending=False).iloc[0]
            cross_prompt_results[qp_name] = best.to_dict()

        # ── Cross-prompt comparison (per model) ─────────────────────────
        if len(cross_prompt_results) > 1:
            print(f"\n{'═' * 60}")
            print(f"Cross-prompt comparison – {model_label} (best config per prompt)")
            print("═" * 60)

            comparison_df = pd.DataFrame(cross_prompt_results).T
            comparison_df.index.name = "query_prompt"
            comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
            print(comparison_df[["f1_macro", "f1_pass", "f1_abstain", "accuracy",
                                 "abstention_rate", "min_top1_relevance",
                                 "min_avg_top3_relevance"]].to_string())
            comparison_df.to_csv(model_outdir / "cross_prompt_comparison.csv")
            print(f"\n  -> Saved to {model_outdir / 'cross_prompt_comparison.csv'}")

    print(f"\nDone. All outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
