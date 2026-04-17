"""
Generate a human annotation spreadsheet for meta-evaluation.

Block 1 – Config comparison (main results):
    3 fixed configs (best / median / worst by mean automated score)
    × all questions = systematic paired comparison.

Block 2 – Metric validation:
    Per-question high-disagreement cases (largest metric spread)
    that are NOT already covered by Block 1.

Exports a CSV pre-populated with question, answer, retrieved context,
and expected answer.

Annotators fill in 3 Likert scales (1-5).
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# ── data paths ──────────────────────────────────────────────────────
PIPELINE_FILE = Path(__file__).resolve().parent.parent / "analysis_output" / "deepeval" / "pipeline_outputs_short_confidential.json"
DEEPEVAL_FILE = Path(__file__).resolve().parent.parent / "analysis_output" / "deepeval" / "deepeval_results_short_confidential.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "human_annotation_sheet.csv"

METRICS = [
    "faithfulness_score",
    "answer_relevancy_score",
    "contextual_relevancy_score",
    "contextual_recall_score",
    "contextual_precision_score",
]

# How many high-disagreement pairs to add in Block 2
BLOCK2_EXTRA = 15


# ── helpers ─────────────────────────────────────────────────────────

def load_scores():
    """Load automated scores keyed by (question, experiment, model_id)."""
    scores = {}
    with open(DEEPEVAL_FILE) as f:
        for row in csv.DictReader(f):
            key = (row["question"], row["experiment"], row["model_id"])
            vals = []
            for m in METRICS:
                try:
                    vals.append(float(row[m]))
                except (ValueError, KeyError):
                    vals.append(None)
            if key not in scores:
                scores[key] = []
            scores[key].append(vals)
    # Average across judge repeats for each key
    averaged = {}
    for key, all_vals in scores.items():
        avg = []
        for i in range(len(METRICS)):
            col_vals = [v[i] for v in all_vals if v[i] is not None]
            avg.append(sum(col_vals) / len(col_vals) if col_vals else 0.0)
        averaged[key] = avg
    return averaged


def load_pipeline_outputs():
    """Load pipeline outputs keyed by (question, experiment, model_id)."""
    with open(PIPELINE_FILE) as f:
        data = json.load(f)
    outputs = {}
    for item in data:
        key = (item["user_input"], item["experiment"], item["model_id"])
        if key not in outputs:
            outputs[key] = item
    return outputs


def pick_fixed_configs(scores):
    """Return (best, median, worst) as (experiment, model_id) tuples by mean score."""
    config_means = defaultdict(list)
    for (q, exp, model), vals in scores.items():
        config_means[(exp, model)].append(sum(vals) / len(vals))
    config_avg = {cfg: sum(v) / len(v) for cfg, v in config_means.items()}
    ranked = sorted(config_avg.items(), key=lambda x: x[1], reverse=True)

    best = ranked[0][0]
    median = ranked[len(ranked) // 2][0]
    worst = ranked[-1][0]
    return best, median, worst


def select_block1(scores, configs):
    """Block 1: all questions × 3 fixed configs."""
    questions = sorted({q for q, _, _ in scores.keys()})
    selected = []
    for q in questions:
        for exp, model in configs:
            key = (q, exp, model)
            if key in scores:
                selected.append((q, exp, model, "block1_config_comparison"))
    return selected


def select_block2(scores, block1_keys, n=BLOCK2_EXTRA):
    """Block 2: highest metric-spread pairs not already in Block 1."""
    candidates = []
    for (q, exp, model), vals in scores.items():
        if (q, exp, model) in block1_keys:
            continue
        spread = max(vals) - min(vals)
        candidates.append((spread, q, exp, model))
    candidates.sort(reverse=True)
    return [(q, exp, model, "block2_metric_validation") for _, q, exp, model in candidates[:n]]


def format_context(contexts, max_chars=1500):
    """Join retrieved context chunks, truncated to max_chars total."""
    parts = []
    total = 0
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx.strip().replace("\n", " ")
        if total + len(snippet) > max_chars:
            snippet = snippet[: max_chars - total] + " [...]"
            parts.append(f"[Chunk {i}] {snippet}")
            break
        parts.append(f"[Chunk {i}] {snippet}")
        total += len(snippet)
    return "\n\n".join(parts)


def build_annotation_csv(selected, pipeline_outputs, scores, configs):
    """Build and write the annotation CSV."""
    rows = []
    for q, exp, model, selection_reason in selected:
        key = (q, exp, model)
        item = pipeline_outputs.get(key)
        if item is None:
            print(f"WARNING: No pipeline output for ({q[:50]}..., {exp}, {model})")
            continue
        rows.append((q, exp, model, selection_reason, item, scores.get(key, [0] * 5)))

    # Randomize order to avoid annotator bias
    random.seed(42)
    random.shuffle(rows)

    # ── CSV columns ─────────────────────────────────────────────────
    header = [
        "annotation_id",
        # Presented to the annotator
        "question",
        "generated_answer",
        # Annotator fills in
        "correctness_1to5",
        "completeness_1to5",
        "usefulness_1to5",
        "notes",
    ]

    # Hidden metadata (for later analysis, annotator can ignore)
    header.extend([
        "_expected_answer",
        "_retrieved_context",
        "_system_abstained",
        "_experiment",
        "_model_id",
        "_query_prompt",
        "_answer_prompt",
        "_temperature",
        "_context_limit",
        "_selection_reason",
        "_auto_faithfulness",
        "_auto_answer_relevancy",
        "_auto_contextual_relevancy",
        "_auto_contextual_recall",
        "_auto_contextual_precision",
        "_gate_confident",
    ])

    csv_rows = []
    for idx, (q, exp, model, reason, item, auto_scores) in enumerate(rows, 1):
        row = {
            "annotation_id": idx,
            "question": q,
            "generated_answer": item["response"],
            # Annotator fills these in
            "correctness_1to5": "",
            "completeness_1to5": "",
            "usefulness_1to5": "",
            "notes": "",
        }

        # Hidden metadata
        row["_expected_answer"] = item["expected_answer"]
        row["_retrieved_context"] = format_context(item.get("retrieved_contexts", []))
        row["_system_abstained"] = item.get("abstained", False)
        row["_experiment"] = exp
        row["_model_id"] = model
        row["_query_prompt"] = item.get("query_prompt", "")
        row["_answer_prompt"] = item.get("answer_prompt", "")
        row["_temperature"] = item.get("temperature", "")
        row["_context_limit"] = item.get("context_limit", "")
        row["_selection_reason"] = reason
        for j, m in enumerate(METRICS):
            row[f"_auto_{m.replace('_score', '')}"] = f"{auto_scores[j]:.3f}"
        row["_gate_confident"] = item.get("confident", "")

        csv_rows.append(row)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write annotator-facing version (no hidden columns)
    annotator_header = [c for c in header if not c.startswith("_")]
    annotator_file = OUTPUT_FILE.with_stem(OUTPUT_FILE.stem + "_annotator")
    with open(annotator_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotator_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nWrote {len(csv_rows)} annotation rows to {OUTPUT_FILE}")
    print(f"Wrote {len(csv_rows)} annotator rows to {annotator_file}  (no hidden columns)")

    # Summary
    b1 = sum(1 for r in csv_rows if r["_selection_reason"].startswith("block1"))
    b2 = sum(1 for r in csv_rows if r["_selection_reason"].startswith("block2"))
    print(f"  Block 1 (config comparison): {b1} rows")
    print(f"  Block 2 (metric validation): {b2} rows")
    print(f"\nSelected configs:")
    for label, (exp, model) in zip(["  Best  ", "  Median", "  Worst "], configs):
        print(f"  {label}: {exp}  [{model}]")

    print(f"\nAnnotator fills in:")
    print(f"  correctness_1to5    – factual accuracy (1=wrong, 5=perfect)")
    print(f"  completeness_1to5   – covers all relevant aspects (1=major gaps, 5=fully complete)")
    print(f"  usefulness_1to5     – would help an enterprise user (1=useless, 5=very helpful)")
    print(f"  notes               – optional free-text comments")
    print(f"\nColumns starting with _ are hidden metadata for analysis.")


def main():
    print("Loading scores...")
    scores = load_scores()
    print(f"  {len(scores)} (question, experiment, model) triples")

    print("Loading pipeline outputs...")
    pipeline_outputs = load_pipeline_outputs()
    print(f"  {len(pipeline_outputs)} unique (question, experiment, model) triples")

    print("Selecting 3 fixed configs (best / median / worst)...")
    best, median, worst = pick_fixed_configs(scores)
    configs = (best, median, worst)

    print("Building Block 1 (config comparison)...")
    block1 = select_block1(scores, configs)
    block1_keys = {(q, exp, model) for q, exp, model, _ in block1}
    print(f"  {len(block1)} pairs")

    print(f"Building Block 2 (metric validation, up to {BLOCK2_EXTRA} extra)...")
    block2 = select_block2(scores, block1_keys)
    print(f"  {len(block2)} pairs")

    selected = block1 + block2
    print(f"\nTotal: {len(selected)} annotation pairs")

    print("Building annotation CSV...")
    build_annotation_csv(selected, pipeline_outputs, scores, configs)


if __name__ == "__main__":
    main()
