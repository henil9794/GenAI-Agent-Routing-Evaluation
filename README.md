# Agentic RAG

An agentic Retrieval-Augmented Generation system with routing between local vector search and web search.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env` and fill in your API keys.

3. Drop domain PDFs into `data/raw_pdfs/`.

4. Index the documents:
   ```bash
   python main.py --mode index
   ```

5. Run evaluation:
   ```bash
   python main.py --mode evaluate
   ```

## Project Structure

- `src/indexer.py` — Chunks and embeds PDFs into a vector store
- `src/retriever.py` — Queries the vector store
- `src/web_search.py` — Tavily web search with caching
- `src/baseline.py` — Naive RAG baseline (no routing)
- `src/agent.py` — Agentic RAG with routing logic
- `src/evaluator.py` — Runs eval dataset and computes metrics
- `data/eval_dataset.csv` — Labeled prompts for evaluation
- `results/` — Output logs and metrics
