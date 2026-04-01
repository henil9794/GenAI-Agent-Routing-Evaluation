from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # NFR: consistent embedding model

def get_embeddings():
    """Shared embeddings instance — always use this, never instantiate separately."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_vectorstore(pdf_dir: str, persist_dir: str = "./chroma_db"):
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")
    return vectorstore

def load_vectorstore(persist_dir: str = "./chroma_db"):
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings()
    )