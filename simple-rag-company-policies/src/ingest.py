from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os

DB_PATH = Path("../chroma_db")
load_dotenv()

def load_document():
    loaders = [
        TextLoader('../data/company-policies.txt')
    ]
    
    original_docs = []
    for loader in loaders:
        original_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
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
    return OpenAIEmbeddings(
        model='text-embedding-3-small',
        openai_api_key=os.getenv('openai_key')
    )

def save_on_chroma(chunks,embedding):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(DB_PATH),
        collection_name="company_policies"
    )
    
    return vectorstore
    