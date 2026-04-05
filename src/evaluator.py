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
    metrics = {"model": model_name, "overall_accuracy": {}}
    for col in tool_cols:
        r_name = col.replace("_tool", "")
        mask = df["ground_truth"] != df[col]
        metrics["overall_accuracy"][r_name] = 1 - (mask.sum() / len(df))
        
        # Stratified by ambiguity tier
        tier_acc = {}
        for tier in [1, 2, 3]:
            tier_mask = df["ambiguity_tier"] == tier
            if tier_mask.any():
                tier_acc[tier] = 1 - (mask[tier_mask].sum() / tier_mask.sum())
        metrics[f"{r_name}_tier_accuracy"] = tier_acc
    return metrics