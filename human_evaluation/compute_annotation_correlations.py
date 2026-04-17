"""
Compute correlations between human annotations and automated metrics,
inter-annotator agreement, and per-config statistical comparisons.

Run AFTER all annotators have filled in human_annotation_sheet.csv.
If multiple annotators, provide files as CLI arguments:
    python compute_annotation_correlations.py annotator1.csv annotator2.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr, kendalltau, wilcoxon
import numpy as np

ANNOTATION_FILE = Path(__file__).resolve().parent / "human_annotation_sheet.csv"

HUMAN_DIMS = [
    "correctness_1to5",
    "completeness_1to5",
    "usefulness_1to5",
]

AUTO_METRICS = [
    ("_auto_faithfulness", "Faithfulness"),
    ("_auto_answer_relevancy", "Answer relevancy"),
    ("_auto_contextual_relevancy", "Contextual relevancy"),
    ("_auto_contextual_recall", "Contextual recall"),
    ("_auto_contextual_precision", "Contextual precision"),
]


def load_annotations(filepath):
    with open(filepath) as f:
        rows = list(csv.DictReader(f))

    results = []
    for r in rows:
        # Skip rows where no human dimension is filled
        if not any(r.get(d, "").strip() for d in HUMAN_DIMS):
            continue

        entry = {"_experiment": r["_experiment"], "_selection_reason": r["_selection_reason"]}
        entry["question"] = r["question"]

        for d in HUMAN_DIMS:
            val = r.get(d, "").strip()
            entry[d] = int(val) if val else None

        for col, _ in AUTO_METRICS:
            entry[col] = float(r[col])

        entry["system_abstained"] = r.get("_system_abstained", "").strip().lower() in ("true", "1", "yes")
        entry["_gate_confident"] = r.get("_gate_confident", "").strip().lower() in ("true", "1", "yes")
        entry["_query_prompt"] = r.get("_query_prompt", "")
        entry["_answer_prompt"] = r.get("_answer_prompt", "")

        results.append(entry)

    return results


# ── 1. Metric-vs-human correlations ────────────────────────────────

def compute_correlations(results):
    print("=" * 80)
    print("METRIC VALIDATION: Automated metric ↔ Human score correlations")
    print("=" * 80)

    for dim in HUMAN_DIMS:
        human_vals = [(r[dim], r) for r in results if r[dim] is not None]
        if len(human_vals) < 5:
            print(f"\n  {dim}: too few annotations ({len(human_vals)}), skipping")
            continue

        human = [h for h, _ in human_vals]
        print(f"\n  Human dimension: {dim}  (n={len(human)})")
        print(f"  {'Automated metric':<25} {'Spearman ρ':>11} {'p':>8} {'Kendall τ':>11} {'p':>8}")
        print(f"  {'-'*70}")

        for col, label in AUTO_METRICS:
            auto = [r[col] for _, r in human_vals]
            rho, p_rho = spearmanr(human, auto)
            tau, p_tau = kendalltau(human, auto)
            sig_r = "*" if p_rho < 0.05 else " "
            sig_t = "*" if p_tau < 0.05 else " "
            print(f"  {label:<25} {rho:>10.3f}{sig_r} {p_rho:>8.4f} {tau:>10.3f}{sig_t} {p_tau:>8.4f}")

    print(f"\n  * = significant at p < 0.05")


# ── 2. Inter-annotator agreement ──────────────────────────────────

def compute_agreement(all_annotator_results):
    """Krippendorff's alpha (ordinal) between annotators."""
    if len(all_annotator_results) < 2:
        print("\n  (Only 1 annotator file – skipping inter-annotator agreement)")
        return

    print("\n" + "=" * 80)
    print("INTER-ANNOTATOR AGREEMENT (Krippendorff's α, ordinal)")
    print("=" * 80)

    # Build a dict: annotation_id -> {annotator_idx: scores}
    for dim in HUMAN_DIMS:
        # Index by annotation_id
        by_id = defaultdict(dict)
        for ann_idx, results in enumerate(all_annotator_results):
            for r in results:
                aid = r.get("question", "") + "|" + r.get("_experiment", "")
                val = r.get(dim)
                if val is not None:
                    by_id[aid][ann_idx] = val

        # Only keep items rated by all annotators
        n_ann = len(all_annotator_results)
        common = {aid: vals for aid, vals in by_id.items() if len(vals) == n_ann}

        if len(common) < 3:
            print(f"  {dim}: too few common annotations ({len(common)}), skipping")
            continue

        # Ordinal Krippendorff's alpha via reliability_data matrix
        n_items = len(common)
        reliability = np.full((n_ann, n_items), np.nan)
        for j, (aid, vals) in enumerate(sorted(common.items())):
            for ann_idx, val in vals.items():
                reliability[ann_idx, j] = val

        alpha = _krippendorff_alpha_ordinal(reliability)
        print(f"  {dim:<25} α = {alpha:.3f}  (n={n_items} items, {n_ann} annotators)")

    print()


def _krippendorff_alpha_ordinal(reliability_data):
    """Compute Krippendorff's alpha for ordinal data."""
    # Flatten valid pairs
    units = []
    for col in range(reliability_data.shape[1]):
        vals = reliability_data[:, col]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 2:
            units.append(valid)

    if not units:
        return 0.0

    # Observed disagreement
    Do = 0.0
    n_pairs = 0
    for vals in units:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                Do += (vals[i] - vals[j]) ** 2
                n_pairs += 1
    if n_pairs == 0:
        return 1.0
    Do /= n_pairs

    # Expected disagreement
    all_vals = np.concatenate(units)
    De = 0.0
    n_total = len(all_vals)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            De += (all_vals[i] - all_vals[j]) ** 2
    De /= (n_total * (n_total - 1) / 2)

    if De == 0:
        return 1.0
    return 1.0 - Do / De


# ── 3. Per-config comparison ──────────────────────────────────────

def compare_configs(results):
    """Wilcoxon signed-rank test between Block 1 configs on same questions."""
    block1 = [r for r in results if r["_selection_reason"].startswith("block1")]
    if not block1:
        return

    print("\n" + "=" * 80)
    print("CONFIG COMPARISON (Block 1 – Wilcoxon signed-rank on paired questions)")
    print("=" * 80)

    # Group by config
    by_config = defaultdict(dict)
    for r in block1:
        by_config[r["_experiment"]][r["question"]] = r

    configs = sorted(by_config.keys())
    if len(configs) < 2:
        print("  Only 1 config in Block 1, skipping comparison.")
        return

    for dim in HUMAN_DIMS:
        print(f"\n  {dim}:")
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                c1, c2 = configs[i], configs[j]
                common_qs = set(by_config[c1].keys()) & set(by_config[c2].keys())
                pairs = []
                for q in sorted(common_qs):
                    v1 = by_config[c1][q].get(dim)
                    v2 = by_config[c2][q].get(dim)
                    if v1 is not None and v2 is not None:
                        pairs.append((v1, v2))
                if len(pairs) < 5:
                    continue
                a, b = zip(*pairs)
                # Extract short config label (query_prompt + answer_prompt)
                label1 = f"{by_config[c1][list(common_qs)[0]].get('_query_prompt', '?')}/{by_config[c1][list(common_qs)[0]].get('_answer_prompt', '?')}"
                label2 = f"{by_config[c2][list(common_qs)[0]].get('_query_prompt', '?')}/{by_config[c2][list(common_qs)[0]].get('_answer_prompt', '?')}"
                try:
                    stat, p = wilcoxon(a, b)
                    sig = "*" if p < 0.05 else " "
                    mean_diff = np.mean(a) - np.mean(b)
                    print(f"    {label1:<30} vs {label2:<30}  Δ={mean_diff:+.2f}  p={p:.4f}{sig}  (n={len(pairs)})")
                except ValueError:
                    print(f"    {label1:<30} vs {label2:<30}  all ties (n={len(pairs)})")


# ── 4. Descriptive stats ──────────────────────────────────────────

def descriptive_stats(results):
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    for dim in HUMAN_DIMS:
        vals = [r[dim] for r in results if r[dim] is not None]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {dim:<25} mean={arr.mean():.2f}  std={arr.std():.2f}  min={arr.min()}  max={arr.max()}  n={len(vals)}")

    # Per query_prompt
    print(f"\n  By query prompt (mean correctness):")
    by_qp = defaultdict(list)
    for r in results:
        if r["correctness_1to5"] is not None:
            by_qp[r.get("_query_prompt", "?")].append(r["correctness_1to5"])
    for qp in sorted(by_qp.keys()):
        vals = by_qp[qp]
        print(f"    {qp:<15} {np.mean(vals):.2f} ± {np.std(vals):.2f}  (n={len(vals)})")

    # Per answer_prompt
    print(f"\n  By answer prompt (mean correctness):")
    by_ap = defaultdict(list)
    for r in results:
        if r["correctness_1to5"] is not None:
            by_ap[r.get("_answer_prompt", "?")].append(r["correctness_1to5"])
    for ap in sorted(by_ap.keys()):
        vals = by_ap[ap]
        print(f"    {ap:<25} {np.mean(vals):.2f} ± {np.std(vals):.2f}  (n={len(vals)})")


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [str(ANNOTATION_FILE)]

    all_annotator_results = []
    for fpath in files:
        results = load_annotations(fpath)
        all_annotator_results.append(results)
        print(f"Loaded {len(results)} annotated rows from {fpath}")

    if not all_annotator_results or not all_annotator_results[0]:
        print("No annotations found. Fill in the annotation sheet first.")
        sys.exit(1)

    # Use first annotator (or merged) for main analysis
    results = all_annotator_results[0]
    if len(results) < 5:
        print(f"Only {len(results)} annotated rows. Need at least 5 for meaningful analysis.")
        sys.exit(1)

    descriptive_stats(results)
    compute_correlations(results)
    compute_agreement(all_annotator_results)
    compare_configs(results)


if __name__ == "__main__":
    main()
