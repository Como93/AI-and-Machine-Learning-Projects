import os 
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#OPENAI_MODEL = 'gpt-4.1'
OLLAMA_MODEL = 'llama3.2'
OLLAMA_BASE_URL = "http://localhost:11434"

TEMPERATURE = 0.3
MAX_TOKENS = 500

SYSTEM_PROMPT = '''
    You are an assistant of company policy. You have to answer only questions about this context.
    
    IMPORTANT RULES:
    - Don't guess information or using external knowledge
    - If the answer doesn't belong the context, reply this: "I don't have any information on this in the company policies"
    - Always quote the section or article number where possible
    - If the context contains partial information, you have to be honest on limits
    - You have to be professional, clear and you only have to reply in Italian Language
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
'''

def get_llm():
    if not os.getenv('openai_key'):
        raise ValueError("openai_key is not available on env file")
    
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        base_url=OLLAMA_BASE_URL
    )
    
    # return ChatOpenAI(
    #     model=OPENAI_MODEL,
    #     temperature=TEMPERATURE,
    #     max_tokens = MAX_TOKENS,
    #     api_key=os.getenv('openai_key')
    # )

def generate_answer(question,context):
    prompt = ChatPromptTemplate.from_messages([
        ("system",SYSTEM_PROMPT),
        ("user","{question}")
    ])
    
    chain = prompt | get_llm() | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "question": question
    })

def generate_answer_with_sources(question,documents):
    context_parts = []
    sources = []
    
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[FONTE {i}]\n{doc.page_content}")
        sources.append({
            "id": i,
            "content_preview": doc.page_content[:100] + "...",
            "source": "company-policies.txt"
        })
    
    context = "---".join(context_parts)
        
    answer = generate_answer(question, context)
    
    return answer, sources
    
    


