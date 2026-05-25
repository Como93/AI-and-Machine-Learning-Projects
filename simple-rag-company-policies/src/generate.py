from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
import re

class Generate:
    def __init__(self,config:Config):
        self.config = config
        
    def get_llm(self):

        return ChatOllama(
            model=self.config.ollama_model,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            base_url=self.config.ollama_base_url
        )
        

    def generate_answer(self,question,context):
        prompt = ChatPromptTemplate.from_messages([
            ("system",self.config.system_prompt),
            ("user","{question}")
        ])
        
        try:
            chain = prompt | self.get_llm() | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response
        except Exception as e:
            print(f"Generating response error {e}")
            return ""
            

    def generate_answer_with_sources(self,question,documents):
        if not documents:
            print(f"No documents found")
            return []
        
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[FONTE {i}]\n{doc.page_content}")
            sources.append({
                "id": i,
                "content_preview": doc.page_content[:100] + "...",
                "source": "company-policies.txt"
            })
        
        context = "\n\n---\n\n".join(context_parts)
            
        answer = self.generate_answer(question, context)
        
        no_context = 'non ho informazioni'
        
        if no_context in answer.lower():
            return answer, []
        
        cited_sources = re.findall(r'FONTE\s*(\d+)', answer)
        cited_sources = list(set(cited_sources)) 
        
        if cited_sources:
            sources = [s for s in sources if str(s["id"]) in cited_sources]
        
        return answer, sources
        
        


