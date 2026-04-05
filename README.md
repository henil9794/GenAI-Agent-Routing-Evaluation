# GenAI Agent Routing Evaluation

A framework for evaluating LLM-based agent routing strategies on financial document queries. The system benchmarks three routing approaches — rule-based, zero-shot LLM, and a LangGraph agentic workflow — across a labeled dataset of prompts with varying ambiguity levels.

---

## Overview

When an agent receives a user query, it must decide whether to answer using:
- **Local retrieval** — a ChromaDB vector store built from 22 company 10-K / annual reports (2025)
- **Web search** — real-time results via the Tavily API

This project evaluates how accurately different routing strategies make that decision, stratified by query ambiguity tier.

### Routing Strategies

| Strategy | Description |
|---|---|
| **Rule-Based** | Keyword matching on temporal cues (`"today"`, `"FY2025"`, etc.) |
| **Zero-Shot LLM** | Direct prompt-based classification with a JSON-structured response |
| **LangGraph Agent** | Multi-node stateful workflow: router → retriever/web_search → synthesizer |

### Models Evaluated

- `meta-llama/llama-3-8b-instruct`
- `openai/gpt-4o-mini`

---

## Project Structure

```
.
├── configs/
│   ├── settings.yaml
│   └── prompts.yaml
├── data/
│   ├── dataset/
│   │   ├── prompts_dataset.json
│   │   └── prompts_dataset_full.json
│   ├── raw/
│   ├── processed/
│   │   └── chroma_db/
│   └── results/
├── notebooks/
│   ├── 01_eda_prompts.ipynb
│   ├── 02_evaluation.ipynb
│   └── 03_visualizations.ipynb
├── proposal/
├── src/
│   ├── agent_router.py
│   ├── baselines.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── tools.py
│   └── utils.py
├── tests/
│   └── test_routers.py
├── main.py
├── requirements.txt
└── run.log
```

---

## Setup

### 1. Clone and create a conda environment

```bash
git clone <repo-url>
cd GenAI-Agent-Routing-Evaluation
conda create -n genai-routing python=3.11 -y
conda activate genai-routing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=<your_openrouter_api_key>
TAVILY_API_KEY=<your_tavily_api_key>
```

The project routes LLM calls through [OpenRouter](https://openrouter.ai), so `OPENROUTER_API_KEY` should be your OpenRouter key.

### 4. (Optional) Adjust settings

Edit `configs/settings.yaml` to change models, chunking parameters, or dataset path.

---

## Running the Evaluation

```bash
python main.py
```

This will:
1. Build the ChromaDB vector store from PDFs in `data/raw/` (first run only)
2. Load the evaluation dataset from `data/dataset/prompts_dataset.json`
3. Run all three routers for each configured model
4. Save per-model results to `data/results/routing_results_<model>.csv`
5. Save metrics (overall and tier-stratified accuracy) to `data/results/metrics_<model>.json`

---

## Dataset

The dataset (`prompts_dataset.json`) contains labeled queries with:

| Field | Description |
|---|---|
| `query_id` | Unique identifier |
| `prompt` | User query |
| `ground_truth` | `"local"` or `"web"` |
| `ambiguity_tier` | 1 (clear) → 3 (ambiguous) |
| `prompt_tags` | Topic tags (e.g., `"revenue"`, `"stock_price"`) |

---

## Results

Output files are saved to `data/results/`:

- `routing_results_<model>.csv` — per-query predictions for all three routers (not part of git)
- `metrics_<model>.json` — overall and tier-stratified accuracy
- `qualitative_failures.csv` — misclassified examples for error analysis (not part of git)
- `confusion_matrices.png`, `model_comparison.png`, `accuracy_vs_tier.png` — visualizations

---

## Configuration Reference

**`configs/settings.yaml`**

```yaml
llm:
  models:
    primary: "meta-llama/llama-3-8b-instruct"
    secondary: "openai/gpt-4o-mini"
  base_url: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"
  temperature: 0.0
  max_tokens: 150

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"

vector_db:
  persist_directory: "./data/processed/chroma_db"
  collection_name: "tech_10k_docs"

tavily:
  api_key_env: "TAVILY_API_KEY"
  max_results: 3

chunking:
  chunk_size: 500
  chunk_overlap: 50
```

---

## Source Documents

22 annual financial reports (2025) are included in `data/raw/`:

Adobe, Airbnb, Alphabet, American Express, Amazon, Apple, Cisco, Dell, DoorDash, GameStop, HP, IBM, Microsoft, Netflix, Pinterest, Qualcomm, Roblox, Salesforce, Snowflake, Uber, Visa, Wayfair.
