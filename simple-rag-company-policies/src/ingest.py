from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os

DB_PATH = Path("../chroma_db")
TEXT = "../data/company-policies.txt"
#EMBEDDING_MODEL = "text-embedding-3-small"
#EMBEDDING_PROVIDER = "openai_key"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL = "http://localhost:11434"
load_dotenv()

def load_document():
    loaders = [
        TextLoader(TEXT)
    ]
    
    original_docs = []
    for loader in loaders:
        original_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n================================================================================\n",
            "\n--------------------------------------------------------------------------------\n",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    return text_splitter.split_documents(original_docs)
    
def create_embedding():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL
    )
    # return OpenAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     openai_api_key=os.getenv(EMBEDDING_PROVIDER)
    # )

def save_on_chroma(chunks,embedding):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(DB_PATH),
        collection_name="company_policies"
    )
    
    return vectorstore
    