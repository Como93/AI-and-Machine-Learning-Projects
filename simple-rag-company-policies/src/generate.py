from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

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
        
        chain = prompt | self.get_llm() | StrOutputParser()
        
        return chain.invoke({
            "context": context,
            "question": question
        })

    def generate_answer_with_sources(self,question,documents):
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
        
        return answer, sources
        
        


