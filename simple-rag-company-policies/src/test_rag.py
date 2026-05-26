from config import Config
from retrieve import Retrieve
from generate import Generate

def test_rag():
    config = Config()
    retrieve = Retrieve(config=config)
    generate = Generate(config=config)
    
    questions = [
        "Quanti giorni di ferie?",
        "Qual è il rimborso chilometrico per le trasferte?",
        "I genitori con figli under 14 quanti giorni di smart working hanno?",
        "Per i corsi di lingua, c'è un rimborso?",
        "Sono previsti buoni pasto?",
        "Come si fa la parmigiana?"
    ]
    
    for question in questions:
        print(question)
        documents = retrieve.retrieve_top_three(question)
        answer, sources = generate.generate_answer_with_sources(question,documents)
        print(answer[:200])
        
        if sources:
            print(f"Fonte: {sources[0]['content_preview'][:60]}...")
        

if __name__ == "__main__":
    test_rag()