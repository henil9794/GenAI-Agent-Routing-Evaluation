import re
import json
from src.utils import get_openrouter_client, load_config, load_prompts

config = load_config()
prompts = load_prompts()

def always_local_router(query: str) -> str:
    return "local"

def rule_based_router(query: str) -> str:
    query_lower = query.lower()
    web_keywords = ["today", "current", "latest", "now", "recent stock price", "breaking", "live"]
    local_keywords = ["202", "fy20", "fiscal", "annual report", "10-k", "past", "historical", "q1", "q2", "q3", "q4"]
    
    if any(kw in query_lower for kw in local_keywords):
        return "local"
    if any(kw in query_lower for kw in web_keywords):
        return "web"
    return "uncertain"

def zero_shot_llm_router(query: str, model_name: str) -> dict:
    client = get_openrouter_client(config)
    prompt = prompts["router_zero_shot"].format(query=query)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"tool": "uncertain", "reason": "LLM JSON parse failed"}

def few_shot_llm_router(query: str, model_name: str) -> dict:
    client = get_openrouter_client(config)
    prompt = prompts["router_few_shot"].format(query=query)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"tool": "uncertain", "reason": "LLM JSON parse failed"}


def cot_llm_router(query: str, model_name: str) -> dict:
    client = get_openrouter_client(config)
    prompt = prompts["router_chain_of_thought"].format(query=query)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"tool": "uncertain", "reason": "LLM JSON parse failed"}
