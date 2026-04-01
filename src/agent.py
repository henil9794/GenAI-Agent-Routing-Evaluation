# import random
# import numpy as np
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, SystemMessage
# from typing import TypedDict, Annotated, Sequence
# from langchain_core.messages import BaseMessage
# import operator

# from src.retriever import local_retrieval
# from src.web_search import web_search

# # NFR04: Fixed random seed for reproducibility
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

# tools = [local_retrieval, web_search]

# # ── FR-14: Prompt Engineering Variants ─────────────────────────────────────
# # Test at least 2 variants and compare their impact on routing accuracy.

# SYSTEM_PROMPTS = {

#     # Variant 1 — Baseline system prompt (minimal instruction)
#     "baseline_prompt": """You are a helpful assistant with access to two tools:
# - local_retrieval: for static, domain-specific knowledge from documents.
# - web_search: for real-time or current information.
# Choose the appropriate tool based on the query.""",

#     # Variant 2 — Chain-of-Thought routing instruction
#     "cot_prompt": """You are a helpful assistant. Before choosing a tool, reason step by step:
# 1. Does this query ask about something time-sensitive or current (news, prices, recent events)?
#    → If yes, use web_search.
# 2. Is this query about static facts, concepts, or content from known documents?
#    → If yes, use local_retrieval.
# 3. If uncertain, ask: does the answer change week to week? If yes → web_search. If no → local_retrieval.
# Available tools:
# - local_retrieval: static domain knowledge from indexed documents.
# - web_search: real-time external information via Tavily.""",

#     # Variant 3 — Explicit ambiguity handling clause
#     "ambiguity_prompt": """You are a routing assistant. Route each query to exactly one tool.
# - local_retrieval: use for definitions, concepts, explanations from static documents.
# - web_search: use for anything with temporal signals (today, current, latest, recent, now, this year).
# AMBIGUITY RULE: If the query contains both a known concept AND a temporal signal
# (e.g., "recent advances in transformers"), prefer web_search since currency matters more.
# Never skip tool use — always call one of the two tools.""",
# }

# # ── Agent Builder ───────────────────────────────────────────────────────────

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     tool_used: str

# def build_agent(model_name: str = "llama3", prompt_variant: str = "baseline_prompt"):
#     """
#     Build a routing agent with a specified LLM and system prompt variant.
#     model_name: 'llama3' or 'mistral'
#     prompt_variant: key from SYSTEM_PROMPTS dict
#     """
#     system_prompt = SYSTEM_PROMPTS[prompt_variant]
#     llm = ChatOllama(model=model_name, temperature=0)  # temp=0 for determinism (Risk mitigation)
#     llm_with_tools = llm.bind_tools(tools)

#     def call_model(state: AgentState):
#         # Prepend system message on the first turn only
#         msgs = state["messages"]
#         if not any(isinstance(m, SystemMessage) for m in msgs):
#             msgs = [SystemMessage(content=system_prompt)] + list(msgs)
#         response = llm_with_tools.invoke(msgs)
#         tool_used = "none"
#         if response.tool_calls:
#             tool_used = response.tool_calls[0]["name"]
#         return {"messages": [response], "tool_used": tool_used}

#     def should_continue(state: AgentState):
#         last = state["messages"][-1]
#         if hasattr(last, "tool_calls") and last.tool_calls:
#             return "tools"
#         return END

#     tool_node = ToolNode(tools)
#     graph = StateGraph(AgentState)
#     graph.add_node("agent", call_model)
#     graph.add_node("tools", tool_node)
#     graph.set_entry_point("agent")
#     graph.add_conditional_edges("agent", should_continue)
#     graph.add_edge("tools", "agent")
#     return graph.compile()


# def run_query(query: str, model_name: str = "llama3",
#               prompt_variant: str = "baseline_prompt") -> dict:
#     agent = build_agent(model_name, prompt_variant)
#     result = agent.invoke({
#         "messages": [HumanMessage(content=query)],
#         "tool_used": "none"
#     })
#     return {
#         "answer": result["messages"][-1].content,
#         "tool_used": result["tool_used"]
#     }

import os, random
import numpy as np
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

from src.retriever import local_retrieval
from src.web_search import web_search

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

tools = [local_retrieval, web_search]

# OpenRouter model name mapping
MODELS = {
    "llama3":  "meta-llama/llama-3-8b-instruct",
    "mistral": "mistralai/mistral-7b-instruct",
}

SYSTEM_PROMPTS = {
    "baseline_prompt": """You are a helpful assistant with access to two tools:
- local_retrieval: for static, domain-specific knowledge from documents.
- web_search: for real-time or current information.
Choose the appropriate tool based on the query.""",

    "cot_prompt": """You are a helpful assistant. Before choosing a tool, reason step by step:
1. Does this query ask about something time-sensitive or current? → web_search
2. Is this query about static facts, concepts, or document content? → local_retrieval
3. If uncertain: does the answer change week to week? Yes → web_search. No → local_retrieval.
Tools available:
- local_retrieval: static domain knowledge from indexed documents.
- web_search: real-time external information via Tavily.""",

    "ambiguity_prompt": """You are a routing assistant. Route each query to exactly one tool.
- local_retrieval: definitions, concepts, explanations from static documents.
- web_search: anything with temporal signals (today, current, latest, recent, now, this year).
AMBIGUITY RULE: If a query has both a known concept AND a temporal signal,
prefer web_search since recency matters more.
Always call one of the two tools — never skip tool use.""",
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_used: str

def build_agent(model_name: str = "llama3", prompt_variant: str = "baseline_prompt"):
    openrouter_model = MODELS.get(model_name, MODELS["llama3"])
    llm = ChatOpenAI(
        model=openrouter_model,
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = SYSTEM_PROMPTS[prompt_variant]

    def call_model(state: AgentState):
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=system_prompt)] + list(msgs)
        response = llm_with_tools.invoke(msgs)
        tool_used = "none"
        if response.tool_calls:
            tool_used = response.tool_calls[0]["name"]
        return {"messages": [response], "tool_used": tool_used}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()

def run_query(query: str, model_name: str = "llama3",
              prompt_variant: str = "baseline_prompt") -> dict:
    agent = build_agent(model_name, prompt_variant)
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
        "tool_used": "none"
    })
    return {"answer": result["messages"][-1].content, "tool_used": result["tool_used"]}