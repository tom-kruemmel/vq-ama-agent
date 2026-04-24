"""Shared RAG pipeline logic used by both the Flask app and evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .agent import RAGAgent
from .confidence_checker import ConfidenceChecker
from .domain_judge import DomainJudge
from .embeddings import Embeddings
from .rank_fusion import RankFusion
from .reranker import CrossEncoderReranker
from .utils import detect_language, sanitize_user_input

logger = logging.getLogger(__name__)


def _chat_history_to_str(chat_history: list[dict[str, str]]) -> str:
    """Convert structured chat history to a string for prompt templates."""
    if not chat_history:
        return "(No previous conversation)"
    lines = []
    for turn in chat_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


@dataclass
class PipelineResult:
    """Structured output from the full RAG pipeline."""

    answer: str | None
    contexts: list[str]
    language: str
    # Domain judge
    domain_in_domain: bool | None = None
    domain_score: float | None = None
    domain_rationale: str | None = None
    rejected_domain: bool = False
    # Confidence gate
    confident: bool | None = None
    confidence_details: dict[str, Any] = field(default_factory=dict)
    abstained: bool = False


def run_pipeline(
    question: str,
    *,
    agent: RAGAgent,
    headings: list[str],
    retriever: Embeddings,
    fusion: RankFusion,
    reranker: CrossEncoderReranker,
    confidence_checker: ConfidenceChecker,
    domain_judge: DomainJudge | None = None,
    rerank_top_n: int = 10,
    top_k: int = 10,
    chat_history: list[dict[str, str]] | None = None,
    enforce_gates: bool = True,
    sanitize: bool = True,
) -> PipelineResult:
    """Execute the full RAG pipeline.

    When *enforce_gates* is True (production), out-of-domain or low-confidence
    queries are rejected with an appropriate message.  When False (evaluation),
    scores are still computed and recorded but the pipeline always proceeds to
    answer generation.
    """
    # 1. Sanitize input
    if sanitize:
        question = sanitize_user_input(question)
        if not question:
            return PipelineResult(answer=None, contexts=[], language="English")

    if chat_history is None:
        chat_history = []

    # 2. Language detection
    language = detect_language(question)

    result = PipelineResult(answer=None, contexts=[], language=language)

    # 3. Domain judge
    if domain_judge is not None:
        chat_history_str = _chat_history_to_str(chat_history)
        in_domain, dq_score, dq_rationale = domain_judge.judge(
            question, chat_history=chat_history_str, min_score=0.60,
        )
        result.domain_in_domain = in_domain
        result.domain_score = dq_score
        result.domain_rationale = dq_rationale
        logger.info(
            "Domain judge: %s (score: %.3f) -- %s", in_domain, dq_score, dq_rationale,
        )
        if not in_domain:
            result.rejected_domain = True
            if enforce_gates:
                if language == "German":
                    result.answer = (
                        "Ich kann bei Fragen zu virtualQ und dessen "
                        "Technologie-Stack helfen. Bitte stellen Sie eine "
                        "entsprechende Frage."
                    )
                else:
                    result.answer = (
                        "I can help with questions about virtualQ and its "
                        "technology stack. Please ask a question related to that."
                    )
                return result

    # 4. Query expansion
    queries = agent.generate_queries(question)

    # 5. Retrieval (dense + BM25)
    retrieved_docs = retriever.retrieve_documents(queries, headings, top_k=top_k)

    # 6. Reciprocal Rank Fusion
    fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)

    # 7. Cross-encoder reranking
    reranked = reranker.rerank(question, fused_docs, top_n=rerank_top_n)

    # 8. Confidence gate
    confident, confidence_details = confidence_checker.evaluate(reranked)
    result.confident = confident
    result.confidence_details = confidence_details

    if not confident:
        result.abstained = True
        logger.info("Abstain gate triggered: %s", confidence_details)
        if enforce_gates:
            if language == "German":
                result.answer = (
                    "Ich habe nicht genügend zuverlässigen Kontext, um diese "
                    "Frage sicher zu beantworten. Bitte formulieren Sie Ihre "
                    "Frage um oder geben Sie mehr Details an."
                )
            else:
                result.answer = (
                    "I don't have enough reliable context to answer that "
                    "confidently. Please rephrase your question or provide a "
                    "bit more detail."
                )
            result.contexts = [chunk.text for chunk in reranked]
            return result

    # 9. Answer generation
    context_texts = [chunk.text for chunk in reranked]
    result.contexts = context_texts
    result.answer = agent.generate_answer(
        question, context_texts, chat_history=chat_history, language=language,
        domain_desc=domain_judge.domain_desc if domain_judge is not None else "",
    )
    return result
