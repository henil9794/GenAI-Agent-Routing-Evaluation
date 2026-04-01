"""
FR-10: Run all prompts in batch against BOTH baseline and agentic systems.
FR-11: Store results in structured CSV/JSON.
FR-12: Compute per-tier accuracy, overall accuracy, precision, recall, F1.
FR-13: Document failure modes.
FR-14: Compare prompt engineering variants.
Logging schema (PRD §6): query_id, prompt_text, ambiguity_tier, ground_truth,
                          predicted_tool, correct, response_text, latency_ms
"""

import time
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.baseline import run_baseline_query
from src.agent import run_query, SYSTEM_PROMPTS

TOOL_CLASSES = ["local_retrieval", "web_search"]


# ── Core batch runner ───────────────────────────────────────────────────────

def run_batch(df: pd.DataFrame, mode: str, model_name: str = "llama3",
              prompt_variant: str = "baseline_prompt") -> pd.DataFrame:
    
    # Drop any rows where prompt_text is missing or not a string
    df = df.dropna(subset=["prompt_text", "query_id", "ground_truth"])
    df = df[df["prompt_text"].apply(lambda x: isinstance(x, str))]
    df = df.reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        print(f"  [{mode}] {row['query_id']}: {row['prompt_text'][:60]}...")
        # ... rest of the function stays exactly the same
        t0 = time.time()

        if mode == "baseline":
            out = run_baseline_query(row["prompt_text"], model_name=model_name)
        else:
            out = run_query(row["prompt_text"], model_name=model_name,
                            prompt_variant=prompt_variant)

        latency_ms = round((time.time() - t0) * 1000)
        predicted = out["tool_used"]
        correct = predicted == row["ground_truth"]

        records.append({
            "query_id":       row["query_id"],
            "prompt_text":    row["prompt_text"],
            "ambiguity_tier": row["ambiguity_tier"],
            "ground_truth":   row["ground_truth"],
            "predicted_tool": predicted,
            "correct":        correct,
            "response_text":  out["answer"],
            "latency_ms":     latency_ms,
            "mode":           mode,
            "model":          model_name,
            "prompt_variant": prompt_variant,
        })
    return pd.DataFrame(records)


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame) -> dict:
    """FR-12: Overall accuracy, per-tier accuracy, precision/recall/F1, degradation rate."""
    y_true = results["ground_truth"]
    y_pred = results["predicted_tool"]

    overall_acc = results["correct"].mean()

    # Per-tier accuracy
    tier_acc = results.groupby("ambiguity_tier")["correct"].mean().to_dict()

    # Ambiguity degradation rate: Tier 1 → Tier 3 drop
    t1 = tier_acc.get(1, None)
    t3 = tier_acc.get(3, None)
    degradation_rate = round((t1 - t3) * 100, 2) if (t1 and t3) else None

    # Precision / Recall / F1 per tool class
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TOOL_CLASSES, zero_division=0
    )
    per_class = {
        cls: {"precision": round(p[i], 3), "recall": round(r[i], 3), "f1": round(f1[i], 3)}
        for i, cls in enumerate(TOOL_CLASSES)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=TOOL_CLASSES)

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_tier_accuracy": {f"tier_{k}": round(v, 4) for k, v in tier_acc.items()},
        "ambiguity_degradation_rate_pct": degradation_rate,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ── Failure mode analysis ────────────────────────────────────────────────────

def analyze_failures(results: pd.DataFrame) -> pd.DataFrame:
    """FR-13: Isolate and return failure cases with linguistic features."""
    failures = results[results["correct"] == False].copy()
    print(f"\n=== Failure Cases ({len(failures)}/{len(results)}) ===")
    for _, row in failures.iterrows():
        print(f"  [{row['query_id']}] Tier {row['ambiguity_tier']} | "
              f"Expected: {row['ground_truth']} | Got: {row['predicted_tool']}")
        print(f"    Query: {row['prompt_text']}")
    return failures


# ── Prompt engineering delta ─────────────────────────────────────────────────

def compute_prompt_delta(results_by_variant: dict) -> dict:
    """
    FR-14: Compare accuracy across prompt variants.
    results_by_variant: { variant_name: results_df }
    Returns accuracy per variant and delta vs baseline_prompt.
    """
    baseline_acc = results_by_variant["baseline_prompt"]["correct"].mean()
    deltas = {}
    for variant, df in results_by_variant.items():
        acc = df["correct"].mean()
        deltas[variant] = {
            "accuracy": round(acc, 4),
            "delta_vs_baseline_pct": round((acc - baseline_acc) * 100, 2)
        }
    return deltas


# ── Main evaluation entry point ──────────────────────────────────────────────

def run_full_evaluation(dataset_path: str, model_name: str = "llama3",
                        output_dir: str = "./results"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    all_results = []

    # 1. Baseline system (FR-10)
    print("\n--- Running BASELINE system ---")
    baseline_df = run_batch(df, mode="baseline", model_name=model_name)
    all_results.append(baseline_df)

    # 2. Agentic system — all 3 prompt variants (FR-14)
    variant_results = {}
    for variant in SYSTEM_PROMPTS:
        print(f"\n--- Running AGENTIC system | model={model_name} | variant={variant} ---")
        result_df = run_batch(df, mode="agentic", model_name=model_name,
                              prompt_variant=variant)
        all_results.append(result_df)
        variant_results[variant] = result_df

    # 3. Save combined log (FR-11)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{output_dir}/routing_log_{model_name}.csv", index=False)
    print(f"\nSaved full log → {output_dir}/routing_log_{model_name}.csv")

    # 4. Compute and save metrics (FR-12)
    report = {}
    report["baseline"] = compute_metrics(baseline_df)
    report["prompt_variants"] = {}
    for variant, vdf in variant_results.items():
        report["prompt_variants"][variant] = compute_metrics(vdf)
    report["prompt_engineering_delta"] = compute_prompt_delta(variant_results)

    with open(f"{output_dir}/metrics_{model_name}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics   → {output_dir}/metrics_{model_name}.json")

    # 5. Print summary
    print("\n========== SUMMARY ==========")
    print(f"Baseline accuracy:  {report['baseline']['overall_accuracy']:.2%}")
    for variant, m in report["prompt_variants"].items():
        delta = report["prompt_engineering_delta"][variant]["delta_vs_baseline_pct"]
        print(f"Agentic [{variant}]: {m['overall_accuracy']:.2%}  (Δ {delta:+.2f}%)")
    print(f"Ambiguity degradation (best variant): "
          f"Tier 1 → Tier 3 = "
          f"{report['prompt_variants']['cot_prompt'].get('ambiguity_degradation_rate_pct', 'N/A')} pp")

    # 6. Failure analysis on best-performing variant
    best_variant = max(variant_results, key=lambda v: variant_results[v]["correct"].mean())
    print(f"\nFailure analysis for best variant: {best_variant}")
    analyze_failures(variant_results[best_variant])

    return report