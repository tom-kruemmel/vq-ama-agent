# src/main.py
import os
from dotenv import load_dotenv
import typer

from .bedrock_client import BedrockClient
from .agent import RAGAgent
from .domain_judge import DomainJudge
from .embeddings import Embeddings
from .rank_fusion import RankFusion
from .pdf_persister import PdfPersister
from .role_assigner_validator import RoleAssignerValidator
from .chat_app import run_chat_server

app = typer.Typer()

def chat_loop(agent: RAGAgent, headings: list[str]):
    typer.echo("Starting chat (type 'exit' to quit)\u2026")
    retriever = Embeddings()
    fusion = RankFusion()
    while True:
        question = typer.prompt("You")
        if question.lower() in ("exit", "quit"):
            break
        queries = agent.generate_queries(question)
        retrieved_docs = retriever.retrieve_documents(queries, headings)
        fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)
        final_answer = agent.generate_answer(question, fused_docs)
        typer.echo(f"Agent: {final_answer}\n")


@app.command()
def cli():
    headings = ["PUBLIC", "CONFIDENTIAL"]
    load_dotenv()
    model_id = os.getenv("BEDROCK_MODEL_ID")
    bedrock = BedrockClient()
    agent = RAGAgent(bedrock, model_id)
    domain_judge = DomainJudge(bedrock, model_id)
    validator = RoleAssignerValidator(directory="data/confluence_pdfs")
    validator.validate_pdfs()
    persister = PdfPersister(directory="data/confluence_pdfs", heading_list=RoleAssignerValidator.heading_list, chunk_size=500, chunk_overlap=100)
    persister.persist_pdfs()
    run_chat_server(agent, domain_judge, headings)

if __name__ == "__main__":
    app()