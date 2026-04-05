import os
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils import load_config, setup_logging

setup_logging()
import logging; logger = logging.getLogger(__name__)

def load_pdfs(raw_dir="./data/raw"):
    docs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".pdf"):
            reader = PdfReader(os.path.join(raw_dir, fname))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            docs.append({"source": fname, "text": text})
    return docs

def build_vector_db(config):
    logger.info("Initializing Vector DB & Ingesting PDFs...")
    docs = load_pdfs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"]
    )
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({"text": chunk, "metadata": {"source": d["source"]}})

    embeddings = HuggingFaceEmbeddings(model_name=config["embeddings"]["model"])
    chroma_client = chromadb.PersistentClient(path=config["vector_db"]["persist_directory"])
    collection = chroma_client.get_or_create_collection(name=config["vector_db"]["collection_name"])

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )
    logger.info(f"Indexed {len(chunks)} chunks into ChromaDB.")
    return collection