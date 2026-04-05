import os
from tavily import TavilyClient
from src.utils import load_config

config = load_config()
_tavily_key_env = config.get("tavily", {}).get("api_key_env", "TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=os.getenv(_tavily_key_env))

def local_retriever(query, collection, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    return [{"text": doc, "meta": meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

def web_searcher(query):
    response = tavily_client.search(query, max_results=config["tavily"]["max_results"])
    return [{"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")} for r in response.get("results", [])]