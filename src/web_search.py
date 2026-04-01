import os
import json
import hashlib
from pathlib import Path
from langchain.tools import tool
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- Result cache (Risk mitigation: avoids rate limits during batch runs) ---
CACHE_FILE = Path("./data/tavily_cache.json")

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()

@tool
def web_search(query: str) -> str:
    """
    Use this for questions requiring real-time or recent information —
    like current events, stock prices, today's news, or anything
    that changes frequently over time.
    """
    key = _cache_key(query)
    cache = _load_cache()

    if key in cache:
        return cache[key]   # return cached result, no API call

    response = client.search(query=query, max_results=4)
    result_text = "\n\n".join([r["content"] for r in response.get("results", [])])

    cache[key] = result_text
    _save_cache(cache)
    return result_text