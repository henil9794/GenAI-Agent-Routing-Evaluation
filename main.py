"""
NFR04: Single entry-point script. Run everything from here.
       Fixed random seed is set inside src/agent.py (SEED=42).
"""

import argparse
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Evaluation Pipeline")
    parser.add_argument("--mode", required=True,
                        choices=["index", "query", "evaluate"],
                        help="Pipeline stage to run")
    parser.add_argument("--query",   type=str, help="Single query (for --mode query)")
    parser.add_argument("--model",   type=str, default="llama3",
                        choices=["llama3", "mistral"], help="Ollama model to use")
    parser.add_argument("--variant", type=str, default="baseline_prompt",
                        choices=["baseline_prompt", "cot_prompt", "ambiguity_prompt"],
                        help="Prompt engineering variant (for --mode query)")
    parser.add_argument("--dataset", type=str, default="data/eval_dataset.csv")
    parser.add_argument("--output",  type=str, default="./results")
    args = parser.parse_args()

    if args.mode == "index":
        from src.indexer import build_vectorstore
        build_vectorstore("data/raw_pdfs/")

    elif args.mode == "query":
        from src.agent import run_query
        result = run_query(args.query, model_name=args.model,
                           prompt_variant=args.variant)
        print(f"\nTool used : {result['tool_used']}")
        print(f"Answer    : {result['answer']}")

    elif args.mode == "evaluate":
        from src.evaluator import run_full_evaluation
        # Run for specified model (run twice manually for llama3 + mistral comparison)
        run_full_evaluation(args.dataset, model_name=args.model,
                            output_dir=args.output)

if __name__ == "__main__":
    main()