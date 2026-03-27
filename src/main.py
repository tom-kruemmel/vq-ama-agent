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
from .role_assigner_validator import RoleAssignerValidator
from .chat_app import run_chat_server

app = typer.Typer()

def chat_loop(agent: RAGAgent, user_roles: list[str], headings: list[str]):
    typer.echo("Starting chat (type ‘exit’ to quit)…")
    while True:
        question = typer.prompt("You")
        if question.lower() in ("exit", "quit"):
            break
        # answer = agent.answer_question(question)
        # agent.answer_from_db(question)
        queries = agent.generate_queries(question)
        retriever = Embeddings()
        retrieved_docs = retriever.retrieve_documents(queries, user_roles, headings)
        fusion = RankFusion()
        fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)
        final_answer = agent.generate_answer(question, fused_docs)
        typer.echo(f"Agent: {final_answer}\n")


@app.command()
def cli():
    user_roles = ["engineer"]  
    headings = ["PUBLIC", "CONFIDENTIAL"]
    load_dotenv()
    model_id = os.getenv("BEDROCK_MODEL_ID")
    #retriever = VectorRetriever(os.getenv("INDEX_PATH", "data/processed/faiss_index.faiss"))
    bedrock = BedrockClient()
    agent = RAGAgent(bedrock, model_id)
    validator = RoleAssignerValidator(directory="data/confluence_pdfs")
    valid, missing, unknown = validator.validate_role_assignments()
    if missing:
        typer.echo(f"Missing role assignments for: {', '.join(missing)}")
    persister = PdfPersister(directory="data/confluence_pdfs", role_map=RoleAssignerValidator.pdf_to_roles_map, heading_list=RoleAssignerValidator.heading_list, chunk_size=500, chunk_overlap=100)
    persister.persist_pdfs()
    # chat_loop(agent,user_roles, headings)
    run_chat_server(agent, user_roles, headings)

if __name__ == "__main__":
    app()