# src/main.py
import os
from dotenv import load_dotenv
import typer

from .bedrock_client import BedrockClient
from .retriever import VectorRetriever
from .agent import RAGAgent

app = typer.Typer()

def chat_loop(agent: RAGAgent):
    typer.echo("Starting chat (type ‘exit’ to quit)…")
    while True:
        question = typer.prompt("You")
        if question.lower() in ("exit", "quit"):
            break
        answer = agent.answer_question(question)
        typer.echo(f"Agent: {answer}\n")

@app.command()
def cli():
    load_dotenv()
    model_id = os.getenv("BEDROCK_MODEL_ID")
    retriever = VectorRetriever(os.getenv("INDEX_PATH", "data/processed/faiss_index.faiss"))
    bedrock = BedrockClient()
    agent = RAGAgent(retriever, bedrock, model_id)

    chat_loop(agent)

if __name__ == "__main__":
    app()