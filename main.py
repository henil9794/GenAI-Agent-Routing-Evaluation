import os
import chromadb
from src.data_loader import build_vector_db
from src.utils import load_config, setup_logging
from src.evaluator import load_dataset, evaluate_routers, compute_metrics, save_json
from src.baselines import rule_based_router, zero_shot_llm_router
from src.agent_router import build_langgraph_agent
import pandas as pd
import logging; logger = logging.getLogger(__name__)

setup_logging()
config = load_config()

def load_collection(config):
    """Load ChromaDB collection, building it first if it doesn't exist."""
    if not os.path.exists(config["vector_db"]["persist_directory"]):
        return build_vector_db(config)
    chroma_client = chromadb.PersistentClient(path=config["vector_db"]["persist_directory"])
    return chroma_client.get_or_create_collection(name=config["vector_db"]["collection_name"])

def main():
    logger.info("Starting Agentic RAG Routing Evaluation")

    # 1. Init Vector DB (run once) and keep the collection handle
    collection = load_collection(config)

    # 2. Load Dataset
    dataset = load_dataset()
    logger.info(f"Loaded {len(dataset)} prompts.")

    # 3. Define Routers
    models = [config["llm"]["models"]["primary"], config["llm"]["models"]["secondary"]]
    for model_name in models:
        logger.info(f"Evaluating model: {model_name}")

        langgraph_agent = build_langgraph_agent(model_name, collection=collection)
        routers = {
            "rule_based": rule_based_router,
            "zero_shot_llm": lambda q: zero_shot_llm_router(q, model_name),
            "langgraph_agent": lambda q: langgraph_agent.invoke({
                "query": q,
                "routing_decision": "",
                "reasoning": "",
                "retrieved_docs": [],
                "final_answer": ""
            })
        }
        
        results_df = evaluate_routers(dataset, model_name, routers)
        metrics = compute_metrics(results_df, model_name)
        
        out_dir = "./data/results"
        os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(f"{out_dir}/routing_results_{model_name.replace('/', '_')}.csv", index=False)
        save_json(metrics, f"{out_dir}/metrics_{model_name.replace('/', '_')}.json")
        logger.info(f"Saved results & metrics for {model_name}")

if __name__ == "__main__":
    main()