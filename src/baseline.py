# # FR-01 to FR-03: Baseline RAG — always retrieves from local vector store only.
# # Used as the comparison system against the agentic router in FR-10.

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from src.indexer import load_vectorstore

# vectorstore = load_vectorstore()
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# llm = ChatOllama(model="llama3", temperature=0)

# prompt_template = ChatPromptTemplate.from_template("""
# You are a helpful assistant. Use the following retrieved context to answer the question.
# If the context does not contain the answer, say "I don't know based on available documents."

# Context:
# {context}

# Question: {question}
# """)

# def run_baseline_query(query: str) -> dict:
#     """
#     Baseline always routes to local_retrieval — no tool selection logic.
#     Returns answer and hardcoded tool label for logging.
#     """
#     docs = retriever.get_relevant_documents(query)
#     context = "\n\n".join([d.page_content for d in docs])
#     chain = prompt_template | llm
#     response = chain.invoke({"context": context, "question": query})
#     return {
#         "answer": response.content,
#         "tool_used": "local_retrieval"   # always local, by definition
#     }

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.indexer import load_vectorstore

# Map shorthand to OpenRouter model names (matches agent.py)
MODELS = {
    "llama3":  "meta-llama/llama-3-8b-instruct",
    "mistral": "mistralai/mistral-7b-instruct",
}

def get_llm(model_name: str = "llama3"):
    """Get LLM instance. model_name can be shorthand ('llama3') or full OpenRouter name."""
    openrouter_model = MODELS.get(model_name, model_name)  # use full name if provided
    return ChatOpenAI(
        model=openrouter_model,
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the retrieved context to answer the question.
If the context does not contain the answer, say "I don't know based on available documents."

Context:
{context}

Question: {question}
""")

def run_baseline_query(query: str, model_name: str = "meta-llama/llama-3-8b-instruct") -> dict:
    docs = retriever.invoke(query)   # ← was retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    chain = PROMPT | get_llm(model_name)
    response = chain.invoke({"context": context, "question": query})
    return {"answer": response.content, "tool_used": "local_retrieval"}