from ingest import Ingest
from retrieve import Retrieve
from generate import Generate
import argparse
from config import Config
import logging


def setup_database(config):
    ingest = Ingest(config)
    chunks = ingest.load_document()
    embeddings = ingest.create_embedding()
    vectorstore = ingest.save_on_chroma(chunks, embeddings)
    
    logging.info(f"Setup Database Completed")
    
    return vectorstore

def ask_question(question,config):
    logging.info(f"Retrieving relevant documents")
    retrieve = Retrieve(config)
    documents = retrieve.retrieve_top_three(question)
    
    if not documents:
        logging.info(f"No documents found")
        return

    generate = Generate(config)
    answer, sources = generate.generate_answer_with_sources(question, documents)
    
    logging.info(f"ANSWER:")
    logging.info(f"{answer}")
    
    logging.info(f"SOURCES")
    for source in sources:
        logging.info(f"\n[Source {source['id']}]")
        logging.info(f"File: {source['source']}")
        logging.info(f"Preview: {source['content_preview']}")
    

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Company Policies")
    parser.add_argument("--setup",action="store_true",help="Setup database")
    parser.add_argument("--question", type=str, help="Single question to answer")
    args = parser.parse_args()
    config = Config()
    logging.basicConfig(filename='CompanyPolicy.log', level=logging.INFO)
    
    if args.setup:
        setup_database(config)
        return
    elif args.question:
        ask_question(args.question,config)
    

    
if __name__ == "__main__":
    main()