from pathlib import Path

class Config:
    def __init__(self):
        self.ollama_model = 'llama3.2'
        self.embedding_model = "nomic-embed-text"
        self.document = Path("../data/company-policies.txt")
        self.ollama_base_url = "http://localhost:11434"
        self.db_path = Path("../chroma_db")
        self.temperature = 0.3
        self.max_tokens = 500
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.top_k = 3
        self.system_prompt = '''
            You are an assistant of company policy. You have to answer only questions about this context.
            
            IMPORTANT RULES:
            - Don't guess information or using external knowledge
            - If the answer doesn't belong the context, reply EXACTLY this phrase: "Non ho informazioni su questo nelle policy aziendali."
            - Always quote the section or article number where possible
            - If the context contains partial information, you have to be honest on limits
            - You have to be professional, clear and you only have to reply in Italian Language
            - At the end of your answer, list which sources you used (es. "Fonti utilizzate: FONTE 1, FONTE 2")
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            ANSWER:
        '''



