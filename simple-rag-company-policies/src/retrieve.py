from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import Config

class Retrieve:
    def __init__(self,config:Config):
        self.config = config

    def load_vectorstore(self):
        if not self.config.db_path.exists() :
            raise FileNotFoundError("Run ingest.py first")
        
        embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url
        )
        
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=self.config.db_path,
            collection_name="company_policies"
        )
        
        return vectorstore

    def retrieve_top_three(self,query: str):
        vectorstore = self.load_vectorstore()
        results = vectorstore.similarity_search(query, k=self.config.top_k)
        return results

    
    