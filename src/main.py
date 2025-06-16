# src/main.py
import os
from dotenv import load_dotenv
import typer

from .bedrock_client import BedrockClient
#from .retriever import VectorRetriever
from .agent import RAGAgent
from .embeddings import Embeddings
from .rank_fusion import RankFusion
from .pdf_persister import PdfPersister

app = typer.Typer()

def chat_loop(agent: RAGAgent):
    typer.echo("Starting chat (type ‘exit’ to quit)…")
    while True:
        question = typer.prompt("You")
        if question.lower() in ("exit", "quit"):
            break
        # answer = agent.answer_question(question)
        # agent.answer_from_db(question)
        queries = agent.generate_queries(question)
        retriever = Embeddings()
        retrieved_docs = retriever.retrieve_documents(queries)
        fusion = RankFusion()
        fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)
        final_answer = agent.generate_answer(question, fused_docs)
        typer.echo(f"Agent: {final_answer}\n")


@app.command()
def cli():
    load_dotenv()
    model_id = os.getenv("BEDROCK_MODEL_ID")
    #retriever = VectorRetriever(os.getenv("INDEX_PATH", "data/processed/faiss_index.faiss"))
    bedrock = BedrockClient()
    agent = RAGAgent(bedrock, model_id)
    persister = PdfPersister(directory="data/confluence_pdfs", chunk_size=1000, chunk_overlap=200)
    persister.persist_pdfs()
    chat_loop(agent)

if __name__ == "__main__":
    app()