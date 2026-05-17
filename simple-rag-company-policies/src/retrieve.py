from pathlib import Path
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

# EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_PROVIDER = "openai_key"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL = "http://localhost:11434"
DB_PATH = Path("../chroma_db")
TOP_K = 3
load_dotenv()

def load_vectorstore():
    if not DB_PATH.exists():
        raise FileNotFoundError("Run ingest.py first")
    
    # embeddings = OpenAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     openai_api_key=os.getenv(EMBEDDING_PROVIDER)
    # )
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL
    )
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH,
        collection_name="company_policies"
    )
    
    return vectorstore

def retrieve_top_three(query: str, top_k: int = TOP_K):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)
    return results

    
    