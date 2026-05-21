from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import Config


class Ingest:
    def __init__(self,config:Config):
        self.config = config
        
    def load_document(self):
        loaders = [
            TextLoader(self.config.document)
        ]
        
        original_docs = []
        for loader in loaders:
            original_docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
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
        
    def create_embedding(self):
        return OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url
        )


    def save_on_chroma(self,chunks,embedding):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=str(self.config.db_path),
            collection_name="company_policies"
        )
        
        return vectorstore
        