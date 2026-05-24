from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import Config

class Retrieve:
    
    _vectorstore_instance = None
    
    def __init__(self,config:Config):
        self.config = config

    def load_vectorstore(self):
        if Retrieve._vectorstore_instance is not None:
            return Retrieve._vectorstore_instance
        
        if not self.config.db_path.exists() :
            raise FileNotFoundError(f"Database not found, run python main.py --setup first")
        
        embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url
        )
        
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=self.config.db_path,
            collection_name="company_policies"
        )
        
        Retrieve._vectorstore_instance = vectorstore

        return vectorstore

    def retrieve_top_three(self,query: str,use_mmr: bool = False):
        if not query:
            print(f"Empty query")
            return []
        
        if len(query) > 500:
            print(f"Query too long, truncate to 500")
            query = query[:500]
        
        try:
            vectorstore = self.load_vectorstore()
            results = None
            if use_mmr:
                results = vectorstore.max_marginal_relevance_search(query, k=self.config.top_k,lambda_mult=0.5)
            else:
                results = vectorstore.similarity_search(query, k=self.config.top_k)
            return results
        except Exception as e:
            print(f"Retrieval error {e}")
            return []
    
    @classmethod
    def clear_vectorestore(cls):
        cls._vectorstore_instance = None

    
    