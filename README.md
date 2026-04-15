# GenAI Agent Routing Evaluation

A framework for evaluating LLM-based agent routing strategies on financial document queries. The system benchmarks six routing approaches - rule-based, always-local, zero-shot LLM, few-shot LLM, chain-of-thought LLM, and a LangGraph agentic workflow - across four LLMs on a labeled dataset of 100 prompts with varying ambiguity levels.

---

## Overview

When an agent receives a user query, it must decide whether to answer using:
- **Local retrieval** - a ChromaDB vector store built from 22 company 10-K / annual reports (2025)
- **Web search** - real-time results via the Tavily API

This project evaluates how accurately different routing strategies make that decision, stratified by query ambiguity tier.

### Routing Strategies

| Strategy | Description |
|---|---|
| **Rule-Based** | Keyword matching on temporal cues (`"today"`, `"FY2025"`, etc.); returns `"uncertain"` if no keyword matches |
| **Always-Local** | Always routes to local retrieval; serves as a lower-bound baseline |
| **Zero-Shot LLM** | Direct prompt-based classification with a JSON-structured response |
| **Few-Shot LLM** | LLM classification with 3 in-context examples |
| **Chain-of-Thought LLM** | Step-by-step reasoning prompt before making a routing decision |
| **LangGraph Agent** | Multi-node stateful workflow: router → retriever/web_search → synthesizer |

### Models Evaluated

| Model | Provider |
|---|---|
| `meta-llama/llama-3-8b-instruct` | Meta |
| `openai/gpt-4o-mini` | OpenAI |
| `google/gemma-3-27b-it` | Google |
| `anthropic/claude-sonnet-4.6` | Anthropic |

All models are queried through [OpenRouter](https://openrouter.ai) with `temperature=0.0` for deterministic routing.

---

## Pipeline

The LangGraph agent follows a 4-node stateful workflow:

```
router_node → route_decision → retriever_node   → synthesizer_node → END
                            ↘ web_search_node ↗
```

- **router_node**: LLM classifies query as `"local"`, `"web"`, or `"uncertain"`
- **retriever_node**: Queries ChromaDB (k=3 documents)
- **web_search_node**: Calls Tavily API (max 3 results)
- **synthesizer_node**: Generates a final answer using retrieved context

A visual diagram is saved to `pipeline.png` after the first run.

---

## Project Structure

```
.
├── configs/
│   ├── settings.yaml           # Models, vector DB, embeddings, chunking config
│   └── prompts.yaml            # Router prompts (zero-shot, few-shot, chain-of-thought)
├── data/
│   ├── dataset/
│   │   └── prompts_dataset.json
│   ├── raw/                    # 22 PDF 10-K / annual reports
│   ├── processed/
│   │   └── chroma_db/          # Persistent vector store (built on first run)
│   └── results/                # CSVs, JSONs, and visualizations from evaluation runs
├── notebooks/
│   ├── 01_eda_prompts.ipynb    # Dataset distribution and tier analysis
│   ├── 02_evaluation.ipynb     # Accuracy metrics and statistical tests
│   └── 03_visualizations.ipynb # Confusion matrices and failure analysis
├── src/
│   ├── agent_router.py         # LangGraph workflow
│   ├── baselines.py            # Rule-based, always-local, and LLM routers
│   ├── data_loader.py          # PDF loading and ChromaDB ingestion
│   ├── evaluator.py            # Metrics computation and failure analysis
│   ├── tools.py                # Local retriever and web searcher tools
│   └── utils.py                # Config loading, OpenRouter client, logging
├── main.py                     # Pipeline orchestrator
├── pipeline.png                # LangGraph workflow visualization
├── requirements.txt
└── run.log
```

---

## Setup

### 1. Clone and create a conda environment

```bash
git clone https://github.com/henil9794/GenAI-Agent-Routing-Evaluation.git
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
3. Run all six routers for each of the four configured models
4. Save per-model results to `data/results/routing_results_<model>.csv`
5. Save metrics (overall and tier-stratified accuracy) to `data/results/metrics_<model>.json`

---

## Dataset

The dataset (`prompts_dataset.json`) contains 100 labeled queries:

| Split | Count |
|---|---|
| Tier 1 - Clear queries | 40 |
| Tier 2 - Moderately ambiguous | 30 |
| Tier 3 - Highly ambiguous | 30 |
| Ground truth: `local` | 52 |
| Ground truth: `web` | 48 |

Each entry includes:

| Field | Description |
|---|---|
| `query_id` | Unique identifier |
| `prompt` | User query |
| `ground_truth` | `"local"` or `"web"` |
| `ambiguity_tier` | 1 (clear) → 3 (ambiguous) |
| `prompt_tags` | Topic tags (e.g., `"revenue"`, `"stock_price"`) |

---

## Results

### Key Findings

- **Rule-based** achieves the highest single-number accuracy (87.8%) but only covers 41% of queries - it abstains on most tier-2 and tier-3 prompts.
- **Claude Sonnet 4.6** is the best-performing LLM backbone, reaching **84% accuracy at 100% coverage** across zero-shot, few-shot, CoT, and LangGraph strategies.
- **Tier 1 (clear queries)** are largely solved by all LLM routers; **Tier 3 (highly ambiguous)** remains the primary challenge, with accuracy dropping to the 56–73% range.
- **Few-shot prompting** provides the largest improvement for mid-tier ambiguity (tier 2), while **CoT** helps on tier 3 for stronger models.
- All LLM-based routers achieve **100% coverage**, making a decision on every query, unlike rule-based.

### Output Files

| File | Description |
|---|---|
| `routing_results_<model>.csv` | Per-query predictions for all six routers |
| `metrics_<model>.json` | Overall and tier-stratified accuracy per router |
| `failure_patterns.csv` | Failure counts aggregated by prompt tag (162 tagged failures) |
| `qualitative_failures.csv` | Misclassified examples with full prompt text and reasoning traces |
| `accuracy_vs_tier.png` | Tier-stratified accuracy bar charts |
| `model_comparison.png` | Cross-model performance comparison |

