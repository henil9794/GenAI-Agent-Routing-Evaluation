"""
Agentic RAG - Labeled Prompt Dataset Generator (Ollama - Free & Local)
Generates local / web / ambiguous labeled prompts directly from a 10-K PDF.

Requirements:
    pip install ollama pymupdf pandas tqdm

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: ollama pull llama3 (or mistral)
    3. Run: python rag_dataset_generator.py
"""

import json
import math
import re
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF
import ollama

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_PATH      = "/Users/parinshah/Documents/NEU/Sem IV/GenAI/Project/MVP/amzn_financial_report_2025.pdf"   # path to your 10-K PDF
OUTPUT_CSV    = "/Users/parinshah/Documents/NEU/Sem IV/GenAI/Project/MVP/rag_eval_dataset.csv"
OUTPUT_JSON   = "/Users/parinshah/Documents/NEU/Sem IV/GenAI/Project/MVP/rag_eval_dataset.json"
TOTAL_PROMPTS = 30                      # total prompts to generate
BATCH_SIZE    = 10                      # prompts per API call (reduced for reliability)
MODEL         = "llama3"                # change to "mistral" if preferred
CHUNK_CHARS   = 4000                    # chars per PDF chunk (reduced for local models)
NUM_CHUNKS    = 3                       # number of chunks to sample from PDF
DEBUG         = True                    # set True to print raw model output
# ───────────────────────────────────────────────────────────────────────────────


# ── Step 1: Extract text from PDF ──────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    full_text = "\n".join(pages)
    print(f"[PDF] Extracted {len(full_text):,} characters from {len(pages)} pages.")
    return full_text


# ── Step 2: Chunk text ─────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_CHARS) -> list[str]:
    """Split text into overlapping chunks so no context is lost at boundaries."""
    overlap = chunk_size // 5
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"[Chunker] Created {len(chunks)} chunks (~{chunk_size:,} chars each).")
    return chunks


# ── Step 3: Pick representative chunks ────────────────────────────────────────
def select_chunks(chunks: list[str], n: int = NUM_CHUNKS) -> str:
    """Select n evenly spaced chunks to cover the whole document."""
    if len(chunks) <= n:
        selected = chunks
    else:
        idxs = [round(i * (len(chunks) - 1) / (n - 1)) for i in range(n)]
        selected = [chunks[i] for i in idxs]
    combined = "\n\n---SECTION BREAK---\n\n".join(selected)
    print(f"[Selector] Using {len(selected)} chunks as context "
          f"({len(combined):,} chars total).")
    return combined


# ── Step 4: Build the prompt ───────────────────────────────────────────────────
def build_prompt(context: str, batch_num: int, batch_size: int,
                 id_offset: int) -> str:
    """
    Single combined prompt for Ollama (no system/user split needed).
    Very explicit instructions to get clean JSON output from local models.
    """
    return f"""You are an expert NLP researcher building evaluation datasets for Agentic RAG systems.

TASK: Given excerpts from a corporate 10-K filing, generate exactly {batch_size} labeled question prompts.

LABEL DEFINITIONS:
- "local"     → Answer is fully contained in the provided 10-K document.
- "web"       → Answer requires real-time information (stock price, recent news, current status).
- "ambiguous" → Routing is genuinely unclear — could be answered by either source.

RULES:
- Write natural, realistic questions a financial analyst would ask.
- Cover diverse topics: revenue, AWS, employees, risk factors, legal, products, stock.
- Ambiguous prompts must use words like "currently", "latest", "now", "still", "recent".
- Include one sentence of reasoning explaining the label.
- Start IDs from {id_offset + 1}.
- Aim for roughly 40% local, 30% web, 30% ambiguous.
- Do NOT include any text outside the JSON array.

OUTPUT FORMAT (strict JSON array only, no markdown, no explanation):
[
  {{
    "id": {id_offset + 1},
    "prompt": "question here",
    "label": "local",
    "reasoning": "one sentence"
  }}
]

--- BEGIN DOCUMENT EXCERPTS (Batch {batch_num}) ---
{context}
--- END DOCUMENT EXCERPTS ---

Generate exactly {batch_size} prompts now. Output only the JSON array:"""


# ── Step 5: Parse JSON robustly ────────────────────────────────────────────────
def parse_json(raw: str) -> list[dict]:
    """
    Try multiple strategies to extract a valid JSON array from model output.
    Local models sometimes wrap output in markdown or add extra text.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: fix trailing commas (common local model issue)
    clean = re.sub(r",\s*([}\]])", r"\1", clean)
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from model output.")


# ── Step 6: Call Ollama ────────────────────────────────────────────────────────
def generate_batch(context: str, batch_num: int, batch_size: int,
                   id_offset: int) -> list[dict]:
    """Call local Ollama model to generate one batch of labeled prompts."""
    prompt = build_prompt(context, batch_num, batch_size, id_offset)

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "num_predict": 3000,   # max tokens to generate
        }
    )

    raw = response["message"]["content"]
    if DEBUG:
        print(f"\n[DEBUG] Raw model output (batch {batch_num}):\n{raw[:800]}\n---")
    return parse_json(raw)


# ── Step 7: Validate each prompt ──────────────────────────────────────────────
def validate(prompts: list[dict]) -> list[dict]:
    """Keep only well-formed prompt entries."""
    valid = []
    for p in prompts:
        if all(k in p for k in ("prompt", "label", "reasoning")):
            if p["label"] in ("local", "web", "ambiguous"):
                if len(p["prompt"].strip()) > 10:
                    valid.append(p)
    removed = len(prompts) - len(valid)
    if removed:
        print(f"[Validate] Removed {removed} malformed prompt(s).")
    return valid


# ── Step 8: Deduplicate ────────────────────────────────────────────────────────
def deduplicate(prompts: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in prompts:
        key = p["prompt"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    removed = len(prompts) - len(unique)
    if removed:
        print(f"[Dedup] Removed {removed} duplicate prompt(s).")
    return unique


# ── Step 9: Save outputs ───────────────────────────────────────────────────────
def save_outputs(prompts: list[dict]) -> None:
    if not prompts:
        print("[ERROR] No prompts to save — all batches failed.")
        print("        Run with DEBUG = True at the top to see raw model output.")
        return
    for i, p in enumerate(prompts, start=1):
        p["id"] = i
    df = pd.DataFrame(prompts)[["id", "label", "prompt", "reasoning"]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Save] CSV  → {OUTPUT_CSV}  ({len(df)} rows)")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"[Save] JSON → {OUTPUT_JSON}  ({len(prompts)} entries)")


# ── Step 10: Print summary ─────────────────────────────────────────────────────
def print_summary(prompts: list[dict]) -> None:
    labels = [p["label"] for p in prompts]
    total  = len(labels)
    print("\n" + "=" * 50)
    print(f"  Dataset Summary ({total} prompts)")
    print("=" * 50)
    for lbl in ["local", "web", "ambiguous"]:
        n   = labels.count(lbl)
        bar = "█" * n
        print(f"  {lbl:<12} {n:>3}  {bar}")
    print("=" * 50)
    print("\nSample prompts:")
    for lbl in ["local", "web", "ambiguous"]:
        sample = next((p for p in prompts if p["label"] == lbl), None)
        if sample:
            print(f"\n  [{lbl.upper()}]\n  Q: {sample['prompt']}\n"
                  f"  → {sample['reasoning']}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Agentic RAG: Labeled Prompt Dataset Generator (Ollama) ===\n")
    print(f"[Config] Model: {MODEL} | Target: {TOTAL_PROMPTS} prompts\n")

    # Check Ollama is running
    try:
        ollama.list()
        print(f"[Ollama] Connected ✓  Using model: {MODEL}\n")
    except Exception:
        print("[ERROR] Ollama is not running. Start it with: ollama serve")
        print("        Then pull your model: ollama pull llama3")
        return

    # 1. Extract and chunk PDF
    text   = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(text)
    ctx    = select_chunks(chunks, n=NUM_CHUNKS)

    # 2. Generate in batches
    n_batches   = math.ceil(TOTAL_PROMPTS / BATCH_SIZE)
    all_prompts: list[dict] = []

    print(f"\n[Gen] Generating {TOTAL_PROMPTS} prompts in "
          f"{n_batches} batches of {BATCH_SIZE}...\n")

    for i in tqdm(range(n_batches), desc="Batches"):
        remaining  = TOTAL_PROMPTS - len(all_prompts)
        this_batch = min(BATCH_SIZE, remaining)
        id_offset  = len(all_prompts)

        try:
            batch = generate_batch(ctx, batch_num=i + 1,
                                   batch_size=this_batch,
                                   id_offset=id_offset)
            batch = validate(batch)
            all_prompts.extend(batch)
            print(f"  Batch {i+1}: +{len(batch)} prompts "
                  f"(total: {len(all_prompts)})")
        except ValueError as e:
            print(f"\n[WARN] Batch {i+1} JSON parse failed: {e}. Skipping.")
        except Exception as e:
            print(f"\n[ERROR] Batch {i+1} failed: {e}")
            break

    # 3. Dedup, save, summarise
    all_prompts = deduplicate(all_prompts)
    save_outputs(all_prompts)
    print_summary(all_prompts)
    print("\n✅ Done! No API costs — ran 100% locally.\n")


if __name__ == "__main__":
    main()