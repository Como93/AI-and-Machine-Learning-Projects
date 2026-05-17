from ingest import load_document,create_embedding,save_on_chroma
from retrieve import retrieve_top_three
from generate import generate_answer_with_sources
import argparse
from pathlib import Path


def setup_database():
    chunks = load_document()
    embeddings = create_embedding()
    vectorstore = save_on_chroma(chunks, embeddings)
    
    print("Database setup complete!")
    
    return vectorstore

def ask_question(question):
    print("Retrieving relevant documents")
    documents = retrieve_top_three(question)
    
    if not documents:
        print("No documents found")
        return

    answer, sources = generate_answer_with_sources(question, documents)
    
    print("ANSWER:")
    print(answer)
    
    print("SOURCES")
    for source in sources:
        print(f"\n[Source {source['id']}]")
        print(f"File: {source['source']}")
        print(f"Preview: {source['content_preview']}")
    

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Company Policies")
    parser.add_argument("--setup",action="store_true",help="Setup database")
    parser.add_argument("--question", type=str, help="Single question to answer")
    args = parser.parse_args()
    
    
    if args.setup:
        setup_database()
        return
    elif args.question:
        ask_question(args.question)
    

    
if __name__ == "__main__":
    main()