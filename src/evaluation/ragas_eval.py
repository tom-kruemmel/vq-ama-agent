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

# === Your app pieces ===
from ..bedrock_client import BedrockClient
from ..agent import RAGAgent
from ..embeddings import Embeddings
from ..rank_fusion import RankFusion

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
        max_retries: int = 2,
        retry_sleep: float = 0.5,
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
        try:
            data = json.loads(text)
            if isinstance(data, dict):
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

        return raw or {}



    # ---- parsing ----
    def _parse_with_schema(self, text: str, schema: Optional[Type[BaseModel]]):
        if schema is None:
            return text

        raw = self._safe_json(text)

        # First attempt: "plain" (strings lists; verdicts as objects)
        data = self._coerce_plain(raw, schema)
        try:
            return schema.model_validate(data)  # pydantic v2
        except Exception:
            try:
                return schema(**data)          # pydantic v1
            except Exception:
                # Second attempt: legacy (dict-wrapped claims/truths; verdicts as strings)
                data2 = self._coerce_legacy(raw, schema)
                try:
                    return schema.model_validate(data2)
                except Exception:
                    return schema(**data2)

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
                    **params,
                )
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.retry_sleep)
        raise last_exc  # pragma: no cover

    # ---- DeepEval LLM interface ----
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        res = self._call_litellm(prompt)
        choice = res.choices[0]
        text = (
            getattr(choice, "message", {}).get("content")
            if hasattr(choice, "message")
            else getattr(choice, "text", "")
        )
        return self._parse_with_schema(text or "", schema)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        res = await self._acall_litellm(prompt)
        choice = res.choices[0]
        text = (
            getattr(choice, "message", {}).get("content")
            if hasattr(choice, "message")
            else getattr(choice, "text", "")
        )
        return self._parse_with_schema(text or "", schema)


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
    last_contexts: List[str] = field(default_factory=list)

    def retrieve(self, question: str, user_roles, headings) -> List[str]:
        queries = self.agent.generate_queries(question)
        retrieved_docs = self.retriever.retrieve_documents(queries, user_roles, headings)
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

    # --------- NEW: define experiments (each with its own questions file) ----------
    experiments = [
        {
            "name": "engineer_default",
            "user_roles": ["engineer"],
            "headings": ["default"],
            "questions_file": "src/evaluation/questions_short.csv",
        },
        {
            "name": "engineer_confidential",
            "user_roles": ["engineer"],
            "headings": ["default","Confidential"],
            "questions_file": "src/evaluation/questions_short.csv",
        },
        # Add more experiments here:
        # {
        #     "name": "manager_default_confidential",
        #     "user_roles": ["manager"],
        #     "headings": ["default", "Confidential"],
        #     "questions_file": "questions_manager.csv",
        # },
    ]
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
    judge_model_id = model_ids[0]
    judge_model = LenientLiteLLMModel(
        model=f"bedrock/{judge_model_id}",
        aws_region_name=region_name,
        temperature=0,
        max_tokens=512,
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

        logger.info(
            f"Running experiment '{exp_name}' with roles={user_roles} "
            f"headings={headings} questions_file={questions_file}"
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

            agent = RAGAgent(my_bedrock_client, gen_model_id)
            app = RAGPipelineApp(agent=agent, retriever=retriever, fusion=fusion)

            test_cases: List[LLMTestCase] = []
            model_outputs = []

            # iterate over questions + expected answers from experiment's CSV
            for item in dataset:
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

                all_rows.append({
                    "experiment": exp_name,
                    "questions_file": questions_file,
                    "model_id": gen_model_id,
                    "question": case.input,
                    "faithfulness_score": faithfulness_score,
                    "answer_relevancy_score": answer_rel_score,
                    "contextual_relevancy_score": contextual_rel_score,
                    "contextual_recall_score": contextual_recall_score,
                    "contextual_precision_score": contextual_precision_score,
                    "num_context_chunks": len(case.retrieval_context or []),
                })
    # ----------------- end experiments loop -----------------

    # Write combined CSV for all models & experiments
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "questions_file",
                "model_id",
                "question",
                "faithfulness_score",
                "answer_relevancy_score",
                "contextual_relevancy_score",
                "contextual_recall_score",
                "contextual_precision_score",
                "num_context_chunks",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Saved DeepEval metric results to {args.csv}")

    # Write combined JSON for all models & experiments
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved pipeline outputs to {args.json}")


