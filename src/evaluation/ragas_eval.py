import json
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm

import time
import asyncio

import os, argparse, logging, json as json_lib
import boto3
from dataclasses import dataclass, field
from dotenv import load_dotenv
from botocore.config import Config
import csv  # <-- moved to top so we can also use it for reading questions.csv

# === logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug logger for judge model outputs (set to DEBUG to see detailed parsing)
judge_logger = logging.getLogger(f"{__name__}.judge_debug")
judge_logger.setLevel(logging.DEBUG)  # Change to INFO to reduce verbosity

# === Your app pieces ===
from ..bedrock_client import BedrockClient
from ..agent import RAGAgent
from ..embeddings import Embeddings
from ..rank_fusion import RankFusion
from ..prompt_templates import (
    # Query prompts
    GENERATE_QUERIES_PROMPT,
    GENERATE_QUERIES_DIVERSE,
    GENERATE_QUERIES_HYDE,
    GENERATE_QUERIES_DECOMPOSE,
    # Answer prompts
    GENERATE_ANSWER_PROMPT,
    #GENERATE_ANSWER_QA_TERSE,
    GENERATE_ANSWER_CONVERSATIONAL,
    GENERATE_ANSWER_INSTRUCTION_BLOCK,
    # GENERATE_ANSWER_STRICT_GROUNDING,
    # GENERATE_ANSWER_SOFT_GROUNDING,
    GENERATE_ANSWER_UNCERTAINTY_AWARE,
    # GENERATE_ANSWER_CONCISE,
    # GENERATE_ANSWER_LIGHT_REASONING,
    GENERATE_ANSWER_FULL_COT,
    # GENERATE_ANSWER_BULLET_SUMMARY,
    # Judge prompts
    JUDGE_QUESTION_DOMAIN_PROMPT,
    JUDGE_DOMAIN_LENIENT,
    JUDGE_DOMAIN_TWO_STAGE,
    JUDGE_DOMAIN_CATEGORY,
)

# === DeepEval imports ===
# pip install deepeval litellm
load_dotenv()
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval import evaluate
from deepeval.models import LiteLLMModel


class LenientLiteLLMModel(DeepEvalBaseLLM):
    """
    LiteLLM (Bedrock) -> DeepEval bridge with robust JSON coercion:
      - Supports DeepEval schemas for Statements, Claims, Truths, Verdicts
      - Handles pydantic v1/v2
      - Returns only the parsed object (NOT a tuple)

    Also:
      - exposes timeout + retry behaviour explicitly
      - uses longer default timeout to reduce Bedrock ReadTimeouts
    """
    def __init__(
        self,
        *,
        model: str,
        aws_region_name: str,
        timeout: int = 300,         # <-- default judge timeout
        max_retries: int = 5,       # <-- increased for rate limits
        retry_sleep: float = 2.0,   # <-- increased base sleep for backoff
        **kwargs,
    ):
        self.model = model
        self.aws_region_name = aws_region_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        # Store whatever DeepEval / caller passes (may already include "timeout")
        self.kwargs: Dict[str, Any] = kwargs

        self.load_model()

    # ---- DeepEvalBaseLLM required ----
    def load_model(self) -> None:
        return None

    def get_model_name(self) -> str:
        return self.model

    # ---- helpers ----
    def _safe_json(self, text: str) -> Dict[str, Any]:
        judge_logger.debug(f"[_safe_json] Raw text input (first 500 chars): {text[:500]}")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                judge_logger.debug(f"[_safe_json] Parsed JSON keys: {list(data.keys())}")
                judge_logger.debug(f"[_safe_json] Parsed JSON content: {json.dumps(data, indent=2)[:1000]}")
                return data
            else:
                judge_logger.warning(f"[_safe_json] JSON parsed but not a dict, got {type(data).__name__}")
        except Exception as e:
            judge_logger.warning(f"[_safe_json] JSON parse failed: {e}")
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    if isinstance(data, dict):
                        judge_logger.debug(f"[_safe_json] Extracted JSON from code block: {list(data.keys())}")
                        return data
                except Exception:
                    pass
        return {}

    def _pd_fields(self, schema: Type[BaseModel]) -> set:
        try:
            return set(getattr(schema, "model_fields").keys())  # pydantic v2
        except Exception:
            return set(getattr(schema, "__fields__", {}).keys())  # pyd v1

    def _wrap_plain_list_of_str(self, values) -> list:
        """Normalize arbitrary structure to a list[str]."""
        out = []
        if isinstance(values, list) and values:
            for v in values:
                if isinstance(v, str):
                    out.append(v)
                elif isinstance(v, dict):
                    for k in ("statement", "truth", "claim", "text", "item"):
                        if isinstance(v.get(k), str):
                            out.append(v[k])
                            break
        elif isinstance(values, str):
            out = [values]
        if not out:
            out = [""]
        return out

    def _coerce_verdicts_obj(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preferred for many DeepEval builds:

        verdicts: List[{
            statement: str,
            verdict: Literal['yes', 'no', 'idk'],
            reason: str
        }]

        - If the model already returns 'statement' or 'reason', we preserve them.
        - Otherwise we fill them with empty strings so Pydantic is satisfied.
        """
        # Try multiple possible locations to be robust to model outputs
        src = (
            raw.get("verdicts")
            or raw.get("items")
            or raw.get("statements")
            or raw.get("truths")
            or raw.get("claims")
        )

        out: List[Dict[str, Any]] = []

        if isinstance(src, list):
            for v in src:
                verdict_val = "idk"
                statement_val = ""
                reason_val = ""

                if isinstance(v, dict):
                    # pick up statement-like field
                    for sk in ("statement", "text", "item", "chunk"):
                        if isinstance(v.get(sk), str):
                            statement_val = v[sk]
                            break

                    # pick up reason-like field if present
                    if isinstance(v.get("reason"), str):
                        reason_val = v["reason"]
                    elif isinstance(v.get("explanation"), str):
                        # some builds might use 'explanation'
                        reason_val = v["explanation"]

                    # verdict may already be present
                    if "verdict" in v:
                        verdict_val = str(v["verdict"]).lower()
                else:
                    # if it's just a string, treat it as the verdict
                    verdict_val = str(v).lower()

                if verdict_val not in ("yes", "no", "idk"):
                    verdict_val = "idk"

                out.append(
                    {
                        "statement": statement_val,
                        "verdict": verdict_val,
                        "reason": reason_val,
                    }
                )

        elif isinstance(raw.get("verdict"), str):
            verdict_val = raw["verdict"].lower()
            if verdict_val not in ("yes", "no", "idk"):
                verdict_val = "idk"

            statement_val = raw.get("statement") or ""
            reason_val = raw.get("reason") or ""
            out = [
                {
                    "statement": statement_val,
                    "verdict": verdict_val,
                    "reason": reason_val,
                }
            ]

        if not out:
            out = [
                {
                    "statement": "",
                    "verdict": "no",
                    "reason": "",
                }
            ]

        return {"verdicts": out}



    def _coerce_verdicts_str(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback for builds that expect verdicts: List[str]
        """
        src = raw.get("verdicts")
        out = []
        if isinstance(src, list):
            for v in src:
                if isinstance(v, dict) and "verdict" in v:
                    val = str(v["verdict"]).lower()
                else:
                    val = str(v).lower()
                if val not in ("yes", "no", "idk"):
                    val = "idk"
                out.append(val)
        elif isinstance(raw.get("verdict"), str):
            val = raw["verdict"].lower()
            if val not in ("yes", "no", "idk"):
                val = "idk"
            out = [val]
        if not out:
            out = ["no"]
        return {"verdicts": out}

    # ---- schema coercion ----
    def _coerce_plain(self, raw: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Plain coercion: statements/claims/truths -> List[str]; verdicts -> List[dict]{verdict: ...}
        """
        fields = self._pd_fields(schema)

        if "statements" in fields:
            stmts = (
                raw.get("statements")
                or raw.get("claims")
                or raw.get("truths")
                or raw.get("items")
                or raw.get("text")
            )
            return {"statements": self._wrap_plain_list_of_str(stmts)}

        if "truths" in fields:
            truths = (
                raw.get("truths")
                or raw.get("claims")
                or raw.get("statements")
                or raw.get("items")
                or raw.get("text")
            )
            return {"truths": self._wrap_plain_list_of_str(truths)}

        if "claims" in fields:
            claims = (
                raw.get("claims")
                or raw.get("truths")
                or raw.get("statements")
                or raw.get("items")
                or raw.get("text")
            )
            return {"claims": self._wrap_plain_list_of_str(claims)}

        if "verdicts" in fields:
            return self._coerce_verdicts_obj(raw)

        # Handle schemas with a top-level 'reason' field (e.g., ContextualRelevancyScoreReason)
        if "reason" in fields:
            reason_val = raw.get("reason") or raw.get("explanation") or raw.get("rationale") or ""
            return {"reason": reason_val}

        return raw or {}

    def _coerce_legacy(self, raw: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Legacy coercion: fallback shapes used by some older DeepEval builds.
        - truths/claims as list[dict]
        - verdicts as list[dict]{statement: ..., verdict: ...}
        """
        fields = self._pd_fields(schema)

        if "truths" in fields:
            truths = (
                raw.get("truths")
                or raw.get("claims")
                or raw.get("statements")
                or raw.get("items")
                or raw.get("text")
            )
            return {"truths": [{"truth": t} for t in self._wrap_plain_list_of_str(truths)]}

        if "claims" in fields:
            claims = (
                raw.get("claims")
                or raw.get("truths")
                or raw.get("statements")
                or raw.get("items")
                or raw.get("text")
            )
            return {"claims": [{"claim": c} for c in self._wrap_plain_list_of_str(claims)]}

        if "verdicts" in fields:
            # IMPORTANT: use object-style verdicts (with `statement`)
            return self._coerce_verdicts_obj(raw)

        if "statements" in fields:
            stmts = (
                raw.get("statements")
                or raw.get("claims")
                or raw.get("truths")
                or raw.get("items")
                or raw.get("text")
            )
            return {"statements": self._wrap_plain_list_of_str(stmts)}

        # Handle schemas with a top-level 'reason' field (e.g., ContextualRelevancyScoreReason)
        if "reason" in fields:
            reason_val = raw.get("reason") or raw.get("explanation") or raw.get("rationale") or ""
            return {"reason": reason_val}

        return raw or {}



    # ---- parsing ----
    def _parse_with_schema(self, text: str, schema: Optional[Type[BaseModel]]):
        if schema is None:
            return text

        schema_name = schema.__name__ if schema else "None"
        judge_logger.debug(f"[_parse_with_schema] Schema: {schema_name}")
        judge_logger.debug(f"[_parse_with_schema] Expected fields: {self._pd_fields(schema)}")

        raw = self._safe_json(text)
        
        if not raw:
            judge_logger.error(f"[_parse_with_schema] EMPTY JSON from judge! Raw text: {text[:500]}")

        # First attempt: "plain" (strings lists; verdicts as objects)
        data = self._coerce_plain(raw, schema)
        judge_logger.debug(f"[_parse_with_schema] After _coerce_plain: {json.dumps(data, default=str)[:500]}")
        try:
            result = schema.model_validate(data)  # pydantic v2
            judge_logger.debug(f"[_parse_with_schema] SUCCESS with plain coercion (pydantic v2): {result}")
            return result
        except Exception as e1:
            judge_logger.debug(f"[_parse_with_schema] Plain coercion pydantic v2 failed: {e1}")
            try:
                result = schema(**data)          # pydantic v1
                judge_logger.debug(f"[_parse_with_schema] SUCCESS with plain coercion (pydantic v1): {result}")
                return result
            except Exception as e2:
                judge_logger.debug(f"[_parse_with_schema] Plain coercion pydantic v1 failed: {e2}")
                # Second attempt: legacy (dict-wrapped claims/truths; verdicts as strings)
                data2 = self._coerce_legacy(raw, schema)
                judge_logger.debug(f"[_parse_with_schema] After _coerce_legacy: {json.dumps(data2, default=str)[:500]}")
                try:
                    result = schema.model_validate(data2)
                    judge_logger.debug(f"[_parse_with_schema] SUCCESS with legacy coercion (pydantic v2): {result}")
                    return result
                except Exception as e3:
                    judge_logger.debug(f"[_parse_with_schema] Legacy coercion pydantic v2 failed: {e3}")
                    result = schema(**data2)
                    judge_logger.warning(f"[_parse_with_schema] FALLBACK with legacy coercion (pydantic v1): {result}")
                    return result

    # ---- helpers for empty response detection ----
    def _is_empty_response(self, text: str) -> bool:
        """
        Detect if the judge returned an empty or effectively empty response.
        This includes:
        - Empty string or whitespace only
        - Empty JSON object '{}'
        - JSON with only empty values
        """
        if not text or not text.strip():
            return True
        stripped = text.strip()
        if stripped == '{}':
            return True
        # Check if it's JSON with only empty arrays/strings
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                # Empty dict
                if not data:
                    return True
                # Dict with all empty values
                for v in data.values():
                    if v and v != [''] and v != '':
                        return False
                return True
        except Exception:
            pass
        return False

    def _extract_response_text(self, res) -> str:
        """Extract text content from LiteLLM response."""
        choice = res.choices[0]
        return (
            getattr(choice, "message", {}).get("content")
            if hasattr(choice, "message")
            else getattr(choice, "text", "")
        ) or ""

    # ---- internal call helpers (with retry) ----
    def _call_litellm(self, prompt: str):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                # Build params so we don't double-pass timeout
                params = dict(self.kwargs)          # copy, don't mutate original
                params.setdefault("timeout", self.timeout)

                return litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    aws_region_name=self.aws_region_name,
                    drop_params=True,  # Drop unsupported params for Bedrock
                    **params,
                )
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_sleep)
        raise last_exc  # pragma: no cover

    async def _acall_litellm(self, prompt: str):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                params = dict(self.kwargs)
                params.setdefault("timeout", self.timeout)

                return await litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    aws_region_name=self.aws_region_name,
                    drop_params=True,  # Drop unsupported params for Bedrock
                    **params,
                )
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise
                # Use exponential backoff for rate limit errors
                is_rate_limit = "RateLimitError" in type(e).__name__ or "429" in str(e)
                sleep_time = self.retry_sleep * (2 ** attempt) if is_rate_limit else self.retry_sleep
                logger.warning(f"Attempt {attempt + 1} failed: {type(e).__name__}. Retrying in {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
        raise last_exc  # pragma: no cover

    # ---- DeepEval LLM interface ----
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        schema_name = schema.__name__ if schema else "None"
        judge_logger.debug(f"\n{'='*60}")
        judge_logger.debug(f"[generate] Called with schema: {schema_name}")
        judge_logger.debug(f"[generate] Prompt preview (first 300 chars): {prompt[:300]}...")
        
        # Retry loop for empty responses
        max_empty_retries = 3
        for empty_attempt in range(max_empty_retries):
            res = self._call_litellm(prompt)
            text = self._extract_response_text(res)
            
            judge_logger.debug(f"[generate] Raw judge response (attempt {empty_attempt + 1}): {text}")
            
            if not self._is_empty_response(text):
                break
            
            if empty_attempt < max_empty_retries - 1:
                judge_logger.warning(
                    f"[generate] Empty response from judge (attempt {empty_attempt + 1}/{max_empty_retries}), "
                    f"retrying with reinforced prompt..."
                )
                # Add emphasis to get JSON output on retry
                if empty_attempt == 0:
                    prompt = prompt + "\n\nIMPORTANT: You MUST respond with a valid JSON object. Do not return an empty response."
                else:
                    prompt = prompt + f"\n\n[Retry {empty_attempt + 1}] Please provide the JSON response now."
                time.sleep(self.retry_sleep)
            else:
                judge_logger.error(
                    f"[generate] Judge returned empty response after {max_empty_retries} attempts. "
                    f"Schema: {schema_name}. Falling back to default values."
                )
        
        result = self._parse_with_schema(text, schema)
        judge_logger.debug(f"[generate] Final parsed result: {result}")
        judge_logger.debug(f"{'='*60}\n")
        return result

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        schema_name = schema.__name__ if schema else "None"
        judge_logger.debug(f"\n{'='*60}")
        judge_logger.debug(f"[a_generate] Called with schema: {schema_name}")
        judge_logger.debug(f"[a_generate] Prompt preview (first 300 chars): {prompt[:300]}...")
        
        # Retry loop for empty responses
        max_empty_retries = 3
        original_prompt = prompt
        for empty_attempt in range(max_empty_retries):
            res = await self._acall_litellm(prompt)
            text = self._extract_response_text(res)
            
            judge_logger.debug(f"[a_generate] Raw judge response (attempt {empty_attempt + 1}): {text}")
            
            if not self._is_empty_response(text):
                break
            
            if empty_attempt < max_empty_retries - 1:
                judge_logger.warning(
                    f"[a_generate] Empty response from judge (attempt {empty_attempt + 1}/{max_empty_retries}), "
                    f"retrying with reinforced prompt..."
                )
                # Add emphasis to get JSON output on retry
                if empty_attempt == 0:
                    prompt = original_prompt + "\n\nIMPORTANT: You MUST respond with a valid JSON object. Do not return an empty response."
                else:
                    prompt = original_prompt + f"\n\n[Retry {empty_attempt + 1}] Please provide the JSON response now. Return a complete JSON object."
                await asyncio.sleep(self.retry_sleep)
            else:
                judge_logger.error(
                    f"[a_generate] Judge returned empty response after {max_empty_retries} attempts. "
                    f"Schema: {schema_name}. Falling back to default values."
                )
        
        result = self._parse_with_schema(text, schema)
        judge_logger.debug(f"[a_generate] Final parsed result: {result}")
        judge_logger.debug(f"{'='*60}\n")
        return result


# -------------------------
# Helpers
# -------------------------
def _normalize_context_texts(docs) -> List[str]:
    texts: List[str] = []
    if docs is None:
        return texts
    for d in docs:
        if hasattr(d, "page_content"):
            texts.append(str(getattr(d, "page_content")))
            continue
        if isinstance(d, dict):
            for k in ("page_content", "content", "text", "body", "chunk"):
                if k in d and isinstance(d[k], (str, bytes)):
                    texts.append(
                        d[k].decode("utf-8") if isinstance(d[k], bytes) else str(d[k])
                    )
                    break
            else:
                texts.append(repr(d))
                continue
            continue
        if isinstance(d, (list, tuple)) and len(d) > 0:
            candidate = d[0]
            if isinstance(candidate, (str, bytes)):
                texts.append(
                    candidate.decode("utf-8")
                    if isinstance(candidate, bytes)
                    else candidate
                )
            elif hasattr(candidate, "page_content"):
                texts.append(str(candidate.page_content))
            else:
                texts.append(repr(d))
            continue
        if isinstance(d, (str, bytes)):
            texts.append(d.decode("utf-8") if isinstance(d, bytes) else d)
            continue
        texts.append(repr(d))
    return [t.strip() for t in texts if t and t.strip()]


# -------------------------
# RAG app
# -------------------------
@dataclass
class RAGPipelineApp:
    agent: object
    retriever: object
    fusion: object
    top_k: int = 10
    last_contexts: List[str] = field(default_factory=list)

    def retrieve(self, question: str, user_roles, headings) -> List[str]:
        queries = self.agent.generate_queries(question)
        retrieved_docs = self.retriever.retrieve_documents(
            queries, user_roles, headings, top_k=self.top_k
        )
        fused_docs = self.fusion.reciprocal_rank_fusion(retrieved_docs)
        contexts = _normalize_context_texts(fused_docs)
        self.last_contexts = contexts
        return contexts

    def generate(self, question: str, contexts: List[str]) -> str:
        return self.agent.generate_answer(question, contexts)

    def query(self, question: str, user_roles, headings) -> str:
        ctx = self.retrieve(question, user_roles, headings)
        return self.generate(question, ctx)


def pick_accessible_model(session, preferred_id: str, region_name: str) -> str:
    bedrock = session.client("bedrock", region_name=region_name)
    resp = bedrock.list_foundation_models(
        byOutputModality="TEXT", byInferenceType="ON_DEMAND"
    )
    ids = {m["modelId"] for m in resp.get("modelSummaries", [])}
    if preferred_id in ids:
        return preferred_id
    cands = [
        mid
        for mid in ids
        if any(s in mid.lower() for s in ("instruct", "chat", "claude", "llama", "mistral", "nova", "qwen"))
    ]
    if cands:
        return sorted(cands)[0]
    if ids:
        return sorted(ids)[0]
    raise RuntimeError("No accessible Bedrock TEXT models found.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG with DeepEval (reference-free metrics)"
    )
    # NOTE: questions file is now defined per experiment (see `experiments` below)
    parser.add_argument("--csv", default="deepeval_results.csv")
    parser.add_argument("--json", default="pipeline_outputs.json")
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "One or more Bedrock model IDs to evaluate (e.g. qwen... nova...). "
            "If omitted, the script will auto-pick a single accessible model."
        ),
    )
    args = parser.parse_args()

    logger.info(
        "Starting DeepEval evaluation (reference-free: faithfulness + answer relevancy + contextual metrics)"
    )

    # --------- Define prompt variants for combinatorial experiments ----------
    QUERY_PROMPTS = {
        "default": GENERATE_QUERIES_PROMPT,
        "diverse": GENERATE_QUERIES_DIVERSE,
        "hyde": GENERATE_QUERIES_HYDE,
        "decompose": GENERATE_QUERIES_DECOMPOSE,
    }

    ANSWER_PROMPTS = {
        "qa_terse": GENERATE_ANSWER_PROMPT,
        "conversational": GENERATE_ANSWER_CONVERSATIONAL,
        "instruction_block": GENERATE_ANSWER_INSTRUCTION_BLOCK,
        "uncertainty_aware": GENERATE_ANSWER_UNCERTAINTY_AWARE,
        "full_cot": GENERATE_ANSWER_FULL_COT,
        # Removed (low discriminative value):
        # "default": GENERATE_ANSWER_PROMPT,
        # "strict_grounding": GENERATE_ANSWER_STRICT_GROUNDING,
        # "soft_grounding": GENERATE_ANSWER_SOFT_GROUNDING,
        # "concise": GENERATE_ANSWER_CONCISE,
        # "light_reasoning": GENERATE_ANSWER_LIGHT_REASONING,
        # "bullet_summary": GENERATE_ANSWER_BULLET_SUMMARY,
    }

    JUDGE_PROMPTS = {
        "default": JUDGE_QUESTION_DOMAIN_PROMPT,
        # "lenient": JUDGE_DOMAIN_LENIENT,
        # "two_stage": JUDGE_DOMAIN_TWO_STAGE,
        # "category": JUDGE_DOMAIN_CATEGORY,
    }

    # --------- Retrieval & generation parameter sweeps ----------
    TOP_K_VALUES = [5, 10, 15]
    TEMPERATURE_VALUES = [0.0, 0.3, 0.7]
    CONTEXT_LIMIT_VALUES = [3, 5, 10]

    # Base configurations for user roles/headings
    BASE_CONFIGS = [
        {
            "user_roles": ["engineer"],
            "headings": ["PUBLIC"],
            "questions_file": "src/evaluation/questions_short_public.csv",
        },
        {
            "user_roles": ["engineer"],
            "headings": ["PUBLIC"],
            "questions_file": "src/evaluation/questions_short_public_de.csv",
        },
        # {
        #     "user_roles": ["engineer"],
        #     "headings": ["PUBLIC", "CONFIDENTIAL"],
        #     "questions_file": "src/evaluation/questions_short_confidential.csv",
        # },
        # {
        #     "user_roles": ["engineer"],
        #     "headings": ["PUBLIC", "CONFIDENTIAL"],
        #     "questions_file": "src/evaluation/questions_short_confidential_de.csv",
        # },
]

    # Generate all combinations of prompts and retrieval/generation params
    experiments = []
    for base_config in BASE_CONFIGS:
        roles_str = "_".join(base_config["user_roles"])
        headings_str = "_".join(h.lower() for h in base_config["headings"])
        
        for query_name, query_prompt in QUERY_PROMPTS.items():
            for answer_name, answer_prompt in ANSWER_PROMPTS.items():
                for judge_name, judge_prompt in JUDGE_PROMPTS.items():
                    for top_k in TOP_K_VALUES:
                        for temperature in TEMPERATURE_VALUES:
                            for context_limit in CONTEXT_LIMIT_VALUES:
                                exp_name = (
                                    f"{roles_str}_{headings_str}"
                                    f"_q_{query_name}_a_{answer_name}_j_{judge_name}"
                                    f"_k{top_k}_t{temperature}_cl{context_limit}"
                                )
                                experiments.append({
                                    "name": exp_name,
                                    "user_roles": base_config["user_roles"],
                                    "headings": base_config["headings"],
                                    "questions_file": base_config["questions_file"],
                                    "generate_queries_prompt": query_prompt,
                                    "generate_answer_prompt": answer_prompt,
                                    "judge_question_domain_prompt": judge_prompt,
                                    "top_k": top_k,
                                    "temperature": temperature,
                                    "context_limit": context_limit,
                                })

    logger.info(f"Generated {len(experiments)} experiment combinations")
    # --------------------------------------------------------------------------    

    region_name = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )
    _ = session.client("bedrock-runtime")

    # Determine which generation model(s) to use
    if args.models:
        model_ids = args.models
        logger.info(f"Using user-specified models: {model_ids}")
    else:
        logger.info("No models specified via --models, trying auto-pick.")
        model_id = pick_accessible_model(
            session,
            preferred_id="qwen.qwen3-235b-a22b-2507-v1:0",
            region_name=region_name,
        )
        if not model_id:
            model_id = pick_accessible_model(
                session,
                preferred_id="eu.amazon.nova-lite-v1:0",
                region_name=region_name,
            )
        if not model_id:
            raise RuntimeError("Could not find an accessible model on Bedrock.")
        model_ids = [model_id]
        logger.info(f"Auto-picked model: {model_id}")

    my_bedrock_client = BedrockClient()
    retriever = Embeddings()
    fusion = RankFusion()

    # Use a single judge model for all systems-under-test
    judge_model_id = "openai.gpt-oss-120b-1:0"
    #judge_model_id = "eu.amazon.nova-pro-v1:0"
    judge_model = LenientLiteLLMModel(
        model=f"bedrock/converse/{judge_model_id}",
        aws_region_name=region_name,
        temperature=0,
        max_tokens=2048,
        timeout=60,
        response_format={"type": "json_object"},
    )

    faithfulness = FaithfulnessMetric(model=judge_model, threshold=0.0)
    answer_rel = AnswerRelevancyMetric(model=judge_model, threshold=0.0)
    contextual_rel = ContextualRelevancyMetric(model=judge_model, threshold=0.0)
    contextual_recall = ContextualRecallMetric(model=judge_model, threshold=0.0)
    contextual_precision = ContextualPrecisionMetric(model=judge_model, threshold=0.0)

    metrics = [
        faithfulness,
        answer_rel,
        contextual_rel,
        contextual_recall,
        contextual_precision,
    ]

    all_outputs = []  # for JSON
    all_rows = []     # for CSV
    completed_keys = set()  # (experiment, model_id, question) tuples already evaluated

    # ---- Resume: load existing results if files exist ----
    CSV_FIELDNAMES = [
        "experiment", "questions_file", "model_id",
        "query_prompt", "answer_prompt", "judge_prompt",
        "top_k", "temperature", "context_limit",
        "question", "faithfulness_score", "answer_relevancy_score",
        "contextual_relevancy_score", "contextual_recall_score",
        "contextual_precision_score", "num_context_chunks",
    ]

    if os.path.isfile(args.csv):
        try:
            with open(args.csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Normalise numeric fields back from strings
                    for score_col in (
                        "faithfulness_score", "answer_relevancy_score",
                        "contextual_relevancy_score", "contextual_recall_score",
                        "contextual_precision_score",
                    ):
                        val = row.get(score_col, "")
                        row[score_col] = float(val) if val not in ("", None) else None
                    nc = row.get("num_context_chunks", "")
                    row["num_context_chunks"] = int(nc) if nc not in ("", None) else 0

                    all_rows.append(row)
                    completed_keys.add(
                        (row["experiment"], row["model_id"], row["question"])
                    )
            logger.info(
                f"Resumed {len(all_rows)} existing rows from {args.csv} "
                f"({len(completed_keys)} unique experiment/model/question combos)"
            )
        except Exception as exc:
            logger.warning(f"Could not load existing CSV for resume ({exc}); starting fresh.")
            all_rows.clear()
            completed_keys.clear()

    if os.path.isfile(args.json):
        try:
            with open(args.json, "r", encoding="utf-8") as f:
                all_outputs = json.load(f)
            logger.info(f"Resumed {len(all_outputs)} existing pipeline outputs from {args.json}")
        except Exception as exc:
            logger.warning(f"Could not load existing JSON for resume ({exc}); starting fresh.")
            all_outputs = []
    # ---- End resume loading ----

    # Helper to load questions for a given experiment
    def load_dataset(path: str):
        dataset = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm_row = {
                    (k or "").strip().lower(): (v or "")
                    for k, v in row.items()
                }
                q = norm_row.get("questions", "").strip()
                expected_ans = norm_row.get("answer", "").strip()
                if not q:
                    continue
                dataset.append(
                    {
                        "question": q,
                        "expected_answer": expected_ans,
                    }
                )
        return dataset

    # ----------------- loop over experiments -----------------
    for experiment in experiments:
        exp_name = experiment["name"]
        user_roles = experiment["user_roles"]
        headings = experiment["headings"]
        questions_file = experiment["questions_file"]
        
        # Extract prompt names for tracking (parse from experiment name or use dedicated fields)
        # Find which prompt variant is being used by matching against the dictionaries
        query_prompt_name = next(
            (k for k, v in QUERY_PROMPTS.items() 
             if v == experiment.get("generate_queries_prompt")), "unknown"
        )
        answer_prompt_name = next(
            (k for k, v in ANSWER_PROMPTS.items() 
             if v == experiment.get("generate_answer_prompt")), "unknown"
        )
        judge_prompt_name = next(
            (k for k, v in JUDGE_PROMPTS.items() 
             if v == experiment.get("judge_question_domain_prompt")), "unknown"
        )
        
        # Extract prompts from experiment config (use defaults if not specified)
        exp_generate_queries_prompt = experiment.get(
            "generate_queries_prompt", GENERATE_QUERIES_PROMPT
        )
        exp_generate_answer_prompt = experiment.get(
            "generate_answer_prompt", GENERATE_ANSWER_PROMPT
        )
        exp_judge_question_domain_prompt = experiment.get(
            "judge_question_domain_prompt", JUDGE_QUESTION_DOMAIN_PROMPT
        )

        # Extract retrieval/generation parameters
        exp_top_k = experiment.get("top_k", 10)
        exp_temperature = experiment.get("temperature", 0.7)
        exp_context_limit = experiment.get("context_limit", 5)

        logger.info(
            f"Running experiment '{exp_name}' with roles={user_roles} "
            f"headings={headings} questions_file={questions_file} "
            f"query_prompt={query_prompt_name} answer_prompt={answer_prompt_name} judge_prompt={judge_prompt_name} "
            f"top_k={exp_top_k} temperature={exp_temperature} context_limit={exp_context_limit}"
        )

        # Load questions for this specific experiment
        try:
            dataset = load_dataset(questions_file)
        except FileNotFoundError:
            logger.error(
                f"Questions file not found for experiment '{exp_name}': {questions_file}. Skipping."
            )
            continue

        if not dataset:
            logger.warning(
                f"No questions found in {questions_file} for experiment '{exp_name}'. Skipping."
            )
            continue

        # Run evaluation for each model separately, per experiment
        for gen_model_id in model_ids:
            logger.info(
                f"Evaluating generation model: {gen_model_id} "
                f"(experiment={exp_name})"
            )

            agent = RAGAgent(
                my_bedrock_client,
                gen_model_id,
                temperature=exp_temperature,
                context_limit=exp_context_limit,
                generate_queries_prompt=exp_generate_queries_prompt,
                generate_answer_prompt=exp_generate_answer_prompt,
                judge_question_domain_prompt=exp_judge_question_domain_prompt,
            )
            app = RAGPipelineApp(
                agent=agent, retriever=retriever, fusion=fusion,
                top_k=exp_top_k,
            )

            test_cases: List[LLMTestCase] = []
            model_outputs = []

            # Check how many questions are already done for this model+experiment
            pending_questions = [
                item for item in dataset
                if (exp_name, gen_model_id, item["question"]) not in completed_keys
            ]
            if not pending_questions:
                logger.info(
                    f"All {len(dataset)} questions already completed for "
                    f"model={gen_model_id} experiment={exp_name}. Skipping."
                )
                continue
            logger.info(
                f"{len(dataset) - len(pending_questions)}/{len(dataset)} questions "
                f"already done; running {len(pending_questions)} remaining "
                f"(model={gen_model_id} experiment={exp_name})"
            )

            # iterate over questions + expected answers from experiment's CSV
            for item in pending_questions:
                q = item["question"]
                expected_ans = item["expected_answer"]

                print(q)
                answer = app.query(q, user_roles, headings)
                contexts = app.last_contexts or []

                logger.info(
                    f"[CASE] experiment={exp_name} model={gen_model_id} "
                    f"question={q[:80]!r} "
                    f"response_len={len(answer) if answer else 0} "
                    f"context_chunks={len(contexts)}"
                )

                model_outputs.append({
                    "experiment": exp_name,
                    "questions_file": questions_file,
                    "model_id": gen_model_id,
                    "user_roles": user_roles,
                    "headings": headings,
                    "query_prompt": query_prompt_name,
                    "answer_prompt": answer_prompt_name,
                    "judge_prompt": judge_prompt_name,
                    "top_k": exp_top_k,
                    "temperature": exp_temperature,
                    "context_limit": exp_context_limit,
                    "user_input": q,
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "expected_answer": expected_ans,
                })

                test_cases.append(
                    LLMTestCase(
                        input=q,
                        actual_output=answer,
                        expected_output=expected_ans or None,
                        retrieval_context=contexts,
                    )
                )
                time.sleep(0.05)

            all_outputs.extend(model_outputs)

            if not test_cases:
                logger.info(f"No new test cases for model={gen_model_id} experiment={exp_name}. Skipping eval.")
                continue

            # Run eval for this model (DeepEval will attach scores to test_cases)
            evaluate(test_cases=test_cases, metrics=metrics)

            # Persist per-question scores (per-model, per-question, per-experiment)
            for case in test_cases:
                logger.info(
                    f"[METRICS] experiment={exp_name} model={gen_model_id} "
                    f"question={case.input[:80]!r} "
                    f"chunks={len(case.retrieval_context or [])}"
                )

                faithfulness_score = None
                answer_rel_score = None
                contextual_rel_score = None
                contextual_recall_score = None
                contextual_precision_score = None

                f = FaithfulnessMetric(model=judge_model, threshold=0.0)
                ar = AnswerRelevancyMetric(model=judge_model, threshold=0.0)
                cr = ContextualRelevancyMetric(model=judge_model, threshold=0.0)
                c_recall = ContextualRecallMetric(model=judge_model, threshold=0.0)
                c_prec = ContextualPrecisionMetric(model=judge_model, threshold=0.0)

                # Faithfulness
                try:
                    f.measure(case)
                    faithfulness_score = getattr(f, "score", None)
                    logger.debug(
                        f"[METRIC OK] Faithfulness "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r} "
                        f"score={faithfulness_score}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[METRIC ERROR] Faithfulness failed for "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r}: {e}"
                    )

                time.sleep(0.15)

                # Answer Relevancy
                try:
                    ar.measure(case)
                    answer_rel_score = getattr(ar, "score", None)
                    logger.debug(
                        f"[METRIC OK] AnswerRelevancy "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r} "
                        f"score={answer_rel_score}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[METRIC ERROR] AnswerRelevancy failed for "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r}: {e}"
                    )

                time.sleep(0.25)

                # Contextual Relevancy
                try:
                    cr.measure(case)
                    contextual_rel_score = getattr(cr, "score", None)
                    logger.debug(
                        f"[METRIC OK] ContextualRelevancy "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r} "
                        f"score={contextual_rel_score}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[METRIC_ERROR] ContextualRelevancy failed for "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r}: {e}"
                    )

                time.sleep(0.25)

                # Contextual Recall
                try:
                    c_recall.measure(case)
                    contextual_recall_score = getattr(c_recall, "score", None)
                    logger.debug(
                        f"[METRIC OK] ContextualRecall "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r} "
                        f"score={contextual_recall_score}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[METRIC ERROR] ContextualRecall failed for "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r}: {e}"
                    )

                time.sleep(0.25)

                # Contextual Precision
                try:
                    c_prec.measure(case)
                    contextual_precision_score = getattr(c_prec, "score", None)
                    logger.debug(
                        f"[METRIC OK] ContextualPrecision "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r} "
                        f"score={contextual_precision_score}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[METRIC_ERROR] ContextualPrecision failed for "
                        f"experiment={exp_name} model={gen_model_id} "
                        f"question={case.input[:80]!r}: {e}"
                    )

                time.sleep(0.25)

                new_row = {
                    "experiment": exp_name,
                    "questions_file": questions_file,
                    "model_id": gen_model_id,
                    "query_prompt": query_prompt_name,
                    "answer_prompt": answer_prompt_name,
                    "judge_prompt": judge_prompt_name,
                    "top_k": exp_top_k,
                    "temperature": exp_temperature,
                    "context_limit": exp_context_limit,
                    "question": case.input,
                    "faithfulness_score": faithfulness_score,
                    "answer_relevancy_score": answer_rel_score,
                    "contextual_relevancy_score": contextual_rel_score,
                    "contextual_recall_score": contextual_recall_score,
                    "contextual_precision_score": contextual_precision_score,
                    "num_context_chunks": len(case.retrieval_context or []),
                }
                all_rows.append(new_row)
                completed_keys.add((exp_name, gen_model_id, case.input))

                # Incremental save after each question so a crash loses at most 1 question
                with open(args.csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(all_rows)
                with open(args.json, "w", encoding="utf-8") as f:
                    json.dump(all_outputs, f, ensure_ascii=False, indent=2)

        # Log progress after each experiment
        logger.info(
            f"Saved intermediate results ({len(all_rows)} total rows) "
            f"to {args.csv} (experiment: {exp_name})"
        )

    # ----------------- end experiments loop -----------------

    # Final save (redundant but confirms completion)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Saved DeepEval metric results to {args.csv} ({len(all_rows)} rows)")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved pipeline outputs to {args.json} ({len(all_outputs)} entries)")

    logger.info("Evaluation complete!")


