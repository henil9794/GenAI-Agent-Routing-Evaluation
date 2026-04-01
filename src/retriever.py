from langchain.tools import tool
from src.indexer import load_vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def local_retrieval(query: str) -> str:
    """
    Use this for questions about static, domain-specific knowledge from
    documents — definitions, concepts, explanations, or historical facts
    that do not change over time.
    """
    docs = retriever.invoke(query)   # ← was retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])