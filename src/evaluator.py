import pandas as pd
import json
from tqdm import tqdm
from src.utils import load_config, save_json
from src.baselines import rule_based_router, zero_shot_llm_router
from src.agent_router import build_langgraph_agent
from sklearn.metrics import accuracy_score, classification_report
import time
CALL_DELAY = 1.0  # seconds between prompts; raise to 2.0 if still stalling
config = load_config()

def load_dataset():
    with open(config["dataset"]["path"], "r") as f:
        return json.load(f)
 
def evaluate_routers(dataset, model_name, routers):
    results = []
    for item in tqdm(dataset, desc=f"Evaluating {model_name}"):
        row = {
            "query_id": item["query_id"],
            "prompt": item["prompt"],
            "ground_truth": item["ground_truth"],
            "ambiguity_tier": item["ambiguity_tier"],
            "prompt_tags": item["prompt_tags"]
        }
        for r_name, r_func in routers.items():
            try:
                pred = r_func(item["prompt"])
                if isinstance(pred, dict):
                    row[f"{r_name}_tool"] = pred.get("tool") or pred.get("routing_decision", "uncertain")
                    row[f"{r_name}_reason"] = pred.get("reasoning", "")
                else:
                    row[f"{r_name}_tool"] = pred
            except Exception as e:
                row[f"{r_name}_tool"] = "error"
                row[f"{r_name}_reason"] = str(e)
        results.append(row)

        time.sleep(CALL_DELAY)  # avoid rate limits
    
    
    return pd.DataFrame(results)

def compute_metrics(df, model_name):
    tool_cols = [c for c in df.columns if c.endswith("_tool")]
    metrics = {"model": model_name, "overall_accuracy": {}, "coverage": {}}
    for col in tool_cols:
        r_name = col.replace("_tool", "")
        decided = df[df[col] != "uncertain"]
        # decided = df[df[col].isin(["local", "web"])]
        if len(decided) > 0:
            mask = decided["ground_truth"] != decided[col]
            metrics["overall_accuracy"][r_name] = 1 - (mask.sum() / len(decided))
            metrics["coverage"][r_name] = len(decided) / len(df)  # fraction that didn't abstain
        else:
            metrics["overall_accuracy"][r_name] = 0.0
            metrics["coverage"][r_name] = 0.0

        # Stratified by ambiguity tier
        tier_acc = {}
        if len(decided) > 0:
            for tier in [1, 2, 3]:
                tier_mask = df["ambiguity_tier"] == tier
                # tier_mask = decided["ambiguity_tier"] == tier
                if tier_mask.any():
                    tier_acc[tier] = 1 - (mask[tier_mask].sum() / tier_mask.sum())
        metrics[f"{r_name}_tier_accuracy"] = tier_acc

        uncertain_rate = {}
        for tier in [1, 2, 3]:
            tier_mask = df["ambiguity_tier"] == tier
            if tier_mask.any():
                uncertain_rate[tier] = (df.loc[tier_mask, col] == "uncertain").sum() / tier_mask.sum()
        metrics[f"{r_name}_uncertain_rate"] = uncertain_rate

    return metrics

def analyze_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each router, find misclassified rows and count failure frequency per prompt_tag.
    Returns a DataFrame: tag x router with failure counts.
    """
    tool_cols = [c for c in df.columns if c.endswith("_tool")]
    tag_failures = {}

    for col in tool_cols:
        r_name = col.replace("_tool", "")
        failed_rows = df[df["ground_truth"] != df[col]]
        tag_counts = {}
        for _, row in failed_rows.iterrows():
            for tag in row["prompt_tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tag_failures[r_name] = tag_counts

    result = pd.DataFrame(tag_failures).fillna(0).astype(int)
    result.index.name = "prompt_tag"
    result = result.sort_values(by=list(result.columns), ascending=False)
    return result

def analyze_qualitative_failures(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    For each router, collect all misclassified rows with prompt text and reasoning.
    Returns a long-form DataFrame: one row per (model, router, failed_prompt).
    """
    tool_cols = [c for c in df.columns if c.endswith("_tool")]
    records = []

    for col in tool_cols:
        r_name = col.replace("_tool", "")
        reason_col = f"{r_name}_reason"
        failed_rows = df[df["ground_truth"] != df[col]]

        for _, row in failed_rows.iterrows():
            records.append({
                "model":          model_name,
                "router":         r_name,
                "query_id":       row["query_id"],
                "prompt":         row["prompt"],
                "ground_truth":   row["ground_truth"],
                "predicted":      row[col],
                "ambiguity_tier": row["ambiguity_tier"],
                "prompt_tags":    row["prompt_tags"],
                "reason":         row.get(reason_col, "")
            })

    return pd.DataFrame(records)
