import json
from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from src.utils import get_openai_client, load_config, load_prompts
from src.tools import local_retriever, web_searcher

config = load_config()
prompts = load_prompts()

class RouterState(TypedDict):
    query: str
    routing_decision: str
    reasoning: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str

def router_node(state: RouterState, model_name: str) -> RouterState:
    client = get_openai_client(config)
    prompt = prompts["router_zero_shot"].format(query=state["query"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    try:
        data = json.loads(response.choices[0].message.content.strip())
        return {**state, "routing_decision": data["tool"], "reasoning": data.get("reason", "")}
    except Exception:
        return {**state, "routing_decision": "uncertain", "reasoning": "Parse fallback"}

def retriever_node(state: RouterState, collection) -> RouterState:
    docs = local_retriever(state["query"], collection)
    return {**state, "retrieved_docs": docs}

def web_search_node(state: RouterState) -> RouterState:
    results = web_searcher(state["query"])
    return {**state, "retrieved_docs": results}

def synthesizer_node(state: RouterState, model_name: str) -> RouterState:
    client = get_openai_client(config)
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {**state, "final_answer": "No relevant information found."}

    context_parts = []
    for doc in docs:
        if "text" in doc:  # local retrieval result
            context_parts.append(doc["text"])
        elif "snippet" in doc:  # web search result
            context_parts.append(f"{doc.get('title', '')}: {doc['snippet']}")

    context = "\n\n".join(context_parts)[:3000]

    synthesis_prompt = (
        f"Answer the following question using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer:"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.1,
        max_tokens=300
    )
    answer = response.choices[0].message.content.strip()
    return {**state, "final_answer": answer}

def route_decision(state: RouterState) -> Literal["local", "web", "uncertain"]:
    decision = state.get("routing_decision", "uncertain")
    if decision not in ("local", "web", "uncertain"):
        return "uncertain"
    return decision

def build_langgraph_agent(model_name: str, collection=None):
    workflow = StateGraph(RouterState)

    workflow.add_node("router", lambda s: router_node(s, model_name))
    workflow.add_node("retriever", lambda s: retriever_node(s, collection))
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("synthesizer", lambda s: synthesizer_node(s, model_name))

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "local": "retriever",
            "web": "web_search",
            "uncertain": "retriever",
        }
    )
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("web_search", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
