import json
import logging
import os
from typing import List

from langchain.prompts import ChatPromptTemplate

from .bedrock_client import BedrockClient
from .prompt_templates import GENERATE_QUERIES_PROMPT, GENERATE_ANSWER_PROMPT, JUDGE_QUESTION_DOMAIN_PROMPT

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Retrieval-Augmented Generation agent using AWS Bedrock as the LLM backend.
    """
    def __init__(
        self,
        bedrock_client: BedrockClient,
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        k: int = 5,
        context_limit: int = 5,
        generate_queries_prompt: str = GENERATE_QUERIES_PROMPT,
        generate_answer_prompt: str = GENERATE_ANSWER_PROMPT,
        judge_question_domain_prompt: str = JUDGE_QUESTION_DOMAIN_PROMPT,
    ):
        self.bedrock = bedrock_client
        self.model_id = model_id
        # Reasoning models (e.g. Qwen3) need a large budget so the
        # chain-of-thought doesn't exhaust all tokens before the answer.
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.k = k
        self.context_limit = context_limit
        self.generate_queries_prompt = generate_queries_prompt
        self.generate_answer_prompt = generate_answer_prompt
        self.judge_question_domain_prompt = judge_question_domain_prompt

    def generate_queries(self, question) -> list[str]:
        prompt = ChatPromptTemplate.from_template(self.generate_queries_prompt)
        raw = self.answer_question(prompt.format_messages(question=question))
        # Split on double-newline first (keeps HyDE passages intact),
        # then split remaining blocks on single newlines (for diverse/decompose).
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        queries = []
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if len(lines) == 1:
                queries.append(lines[0])
            else:
                # Multi-line block: treat the whole block as one query
                # (HyDE passage) rather than splitting into individual lines
                queries.append(block)
        if not queries:
            logger.warning("Query generation returned no queries; falling back to original question.")
        return queries or [question]

    def generate_answer(self, question, context_docs, chat_history: str = "", language: str = "English"):
        context_texts = [doc[0] if isinstance(doc, tuple) else doc for doc in context_docs[:self.context_limit]]
        context = "\n\n".join(context_texts)
        answer_prompt = self.generate_answer_prompt.format(
            context=context, question=question, chat_history=chat_history, language=language
        )
        return self.answer_question(answer_prompt)

    def answer_question(self, question: str) -> str:
        """
        Queries Bedrock to generate an answer.
        """
        response = self.bedrock.invoke_model(
            model_id=self.model_id,
            prompt=question,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        # Some models (e.g. Qwen3) return reasoningContent blocks before the
        # text block.  Walk the list and return the first text entry.
        content_blocks = response['output']['message']['content']
        for block in content_blocks:
            if 'text' in block:
                return block['text']

        # Fallback: if the model exhausted its token budget on reasoning and
        # never produced a text block, extract the reasoning text so we don't
        # crash the pipeline.
        for block in content_blocks:
            rc = block.get('reasoningContent', {})
            rt = rc.get('reasoningText', rc)  # may be nested dict or str
            if isinstance(rt, dict) and 'text' in rt:
                return rt['text']
            if isinstance(rt, str):
                return rt

        raise ValueError(
            f"No 'text' block found in model response content: "
            f"{content_blocks}"
        )

    def judge_question_domain(self, question: str, *, chat_history: str = "", min_score: float = 0.60):
        """
        Uses the LLM to judge whether the user's QUESTION is in-domain for:
        'virtualQ (the company) and its technology stack.'
        Returns (in_domain: bool, score: float, rationale: str).
        """
        domain_desc = (
            "Questions specifically about the telephony company virtualQ and its technology stack. "
            "This includes: virtualQ's products, APIs/SDKs, architecture, cloud providers, "
            "datastores, infrastructure, integrations, deployment/CICD, observability, "
            "security/compliance, and engineering practices at virtualQ. Be sure to include technology related questions in the domain especially if they are related to telephony. Include all questions that are reasonable to ask a technology company. "
            "It excludes unrelated general knowledge and questions about other companies."
        )

        prompt = self.judge_question_domain_prompt.format(
            domain_desc=domain_desc, question=question, chat_history=chat_history
        )

        response = self.bedrock.invoke_model(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
        )
        # Extract the first text block (reasoning models may prepend
        # reasoningContent blocks before the actual text).
        raw = None
        for block in response['output']['message']['content']:
            if 'text' in block:
                raw = block['text'].strip()
                break
        if raw is None:
            raw = str(response['output']['message']['content'])

        try:
            data = json.loads(raw)
            in_domain = bool(data.get("in_domain"))
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Judge JSON parse failed: %s — raw response: %.200s", exc, raw)
            in_domain, score, rationale = False, 0.0, "Judge JSON parse failed."

        # Apply threshold
        in_domain = in_domain and (score >= min_score)
        return in_domain, score, rationale
