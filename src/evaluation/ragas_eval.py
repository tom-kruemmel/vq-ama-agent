# #!/usr/bin/env python3
# """Updated ragas evaluation script using a synchronous Bedrock LLM wrapper with logging.

# Notes:
# - Provides BedrockSyncLLM (LangChain-compatible) that calls boto3 bedrock-runtime.invoke_model
#   synchronously and implements an async wrapper for compatibility with ragas' async usage.
# - Avoids streaming/streaming-parsing issues from langchain_aws for bedrock-mistral.
# - Keep your project's relative imports as they were (BedrockClient, RAGAgent, etc.).

# Usage:
#     python -m evaluation.ragas_eval --questions path/to/questions.txt

# Make sure AWS creds + BEDROCK_MODEL_ID (or default) are set in env.
# """

# import argparse
# import asyncio
# import json
# import os
# from typing import Any, Dict, Optional

# import boto3
# from dotenv import load_dotenv

# load_dotenv()

# import logging
# from logging.handlers import RotatingFileHandler
# import time

# # Configure logging with both console and file handlers
# LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# logger = logging.getLogger("ragas_eval")
# logger.setLevel(logging.DEBUG)

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
# logger.addHandler(console_handler)

# # File handler with rotation
# os.makedirs("logs", exist_ok=True)
# log_path = os.path.join("logs", "ragas_eval.log")
# file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
# logger.addHandler(file_handler)

# logger.info("Logging initialized. Log file at %s", os.path.abspath(log_path))

# # LangChain LLM base import (support multiple langchain packaging names)
# try:
#     from langchain_core.language_models.llms import LLM
# except Exception:
#     try:
#         from langchain.llms.base import LLM
#     except Exception:
#         # Fallback for older/newer variants — raise clear error
#         raise ImportError(
#             "Unable to import LLM base from langchain. Please install a compatible langchain package."
#         )

# # Ragas / project imports
# from ragas.dataset_schema import EvaluationDataset
# from ragas.metrics import faithfulness, answer_relevancy
# from ragas import evaluate

# # keep your project-relative imports (adjust if you moved files)
# from ..bedrock_client import BedrockClient
# from ..agent import RAGAgent
# from ..embeddings import Embeddings
# from ..rank_fusion import RankFusion
# from langchain_aws import BedrockEmbeddings


# class BedrockSyncLLM(LLM):
#     """LangChain-compatible LLM that calls AWS Bedrock synchronously via boto3."""

#     client: Any
#     model_id: str

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         return {"model_id": self.model_id}

#     @property
#     def _llm_type(self) -> str:
#         return "bedrock-sync"

#     def _repair_json(self, text: str) -> Optional[object]:
#         if not isinstance(text, str):
#             return None

#         txt = text.strip()
#         try:
#             return json.loads(txt)
#         except Exception:
#             pass

#         first = txt.find('{')
#         last = txt.rfind('}')
#         if first != -1 and last != -1 and last > first:
#             candidate = txt[first:last+1]
#             try:
#                 return json.loads(candidate)
#             except Exception:
#                 cand = candidate.replace("\n", ' ').replace("'", '"').replace(',}', '}').replace(',]', ']')
#                 try:
#                     return json.loads(cand)
#                 except Exception:
#                     pass

#         if '}{' in txt:
#             parts = txt.split('}{')
#             rebuilt = '[' + ','.join((p if p.strip().startswith('{') else '{'+p) for p in parts) + ']'
#             try:
#                 return json.loads(rebuilt)
#             except Exception:
#                 pass

#         cand2 = txt.replace("'", '"').replace(',}', '}').replace(',]', ']')
#         try:
#             return json.loads(cand2)
#         except Exception:
#             return None

#     def _wrap_for_ragas(self, parsed: Optional[object], text_fallback: Optional[str]) -> str:
#         if isinstance(parsed, dict):
#             for k in ("answer", "final_answer", "response", "output"):
#                 if k in parsed:
#                     return json.dumps(parsed)
#             return json.dumps({"answer": parsed, "sources": parsed.get("sources") if isinstance(parsed.get("sources"), list) else []})

#         if isinstance(parsed, list):
#             return json.dumps({"answer": parsed, "sources": []})

#         safe_text = text_fallback or ""
#         return json.dumps({"answer": safe_text.strip(), "sources": []})

#     def _log_raw_output(self, text: str) -> None:
#         try:
#             logger.debug("Logging raw model output (truncated to 1000 chars): %s", text[:1000])
#             raw_log_path = os.path.join("logs", "ragas_raw_outputs.log")
#             with open(raw_log_path, 'a', encoding='utf-8') as fh:
#                 fh.write('---- RAW OUTPUT START ----\n')
#                 fh.write(text + '\n')
#                 fh.write('---- RAW OUTPUT END ----\n\n')
#             logger.info("Raw model output logged to %s", os.path.abspath(raw_log_path))
#         except Exception as e:
#             logger.exception("Failed to write raw output log: %s", e)

#     def _call(self, prompt: str, *args, **kwargs) -> str:
#         stop = None
#         if 'stop' in kwargs:
#             stop = kwargs.pop('stop')
#         elif len(args) >= 1:
#             stop = args[0]

#         kwargs.setdefault('temperature', 0.0)
#         kwargs.setdefault('max_tokens', 2048)

#         if 'mistral' in self.model_id.lower():
#             body = {
#                 'prompt': prompt,
#                 'max_tokens': kwargs.get('max_tokens', 512),
#                 'temperature': kwargs.get('temperature', 0.0),
#                 'top_p': kwargs.get('top_p', 1.0),
#             }
#         else:
#             body = {'prompt': prompt}

#         payload = json.dumps(body).encode('utf-8')

#         logger.info("Invoking model %s with prompt length=%d, max_tokens=%s", self.model_id, len(prompt), kwargs.get('max_tokens'))

#         response = self.client.invoke_model(
#             modelId=self.model_id,
#             contentType='application/json',
#             accept='application/json',
#             body=payload,
#         )

#         raw = response.get('body')
#         try:
#             raw_bytes = raw.read() if hasattr(raw, 'read') else raw
#             model_response = json.loads(raw_bytes.decode('utf-8'))
#         except Exception:
#             raw_text = str(raw)
#             repaired = self._repair_json(raw_text)
#             if repaired is not None:
#                 return self._wrap_for_ragas(repaired, None)
#             self._log_raw_output(raw_text)
#             return self._wrap_for_ragas(None, raw_text)

#         text = None
#         parsed_candidate = None
#         if isinstance(model_response, dict):
#             parsed_candidate = model_response

#             if 'outputs' in model_response and isinstance(model_response['outputs'], list) and model_response['outputs']:
#                 first = model_response['outputs'][0]
#                 if isinstance(first, dict) and isinstance(first.get('text'), str):
#                     text = first.get('text')
#                 elif isinstance(first, dict) and 'content' in first and isinstance(first['content'], list):
#                     parts = [c.get('text') for c in first['content'] if isinstance(c, dict) and isinstance(c.get('text'), str)]
#                     if parts:
#                         text = ''.join(parts)

#             if text is None and 'outputText' in model_response:
#                 text = model_response.get('outputText')

#             if text is None:
#                 for k in ('generated_text', 'text', 'output', 'completion'):
#                     if k in model_response and isinstance(model_response[k], str):
#                         text = model_response[k]
#                         break

#         if isinstance(text, str):
#             repaired = self._repair_json(text)
#             if repaired is not None:
#                 return self._wrap_for_ragas(repaired, None)
#             return self._wrap_for_ragas(None, text)

#         if parsed_candidate is not None:
#             try:
#                 json.dumps(parsed_candidate)
#                 repaired = parsed_candidate
#             except Exception:
#                 repaired = None

#             if repaired is not None:
#                 for k in ("answer", "final_answer", "response", "output"):
#                     if k in repaired:
#                         return json.dumps(repaired)
#                 return self._wrap_for_ragas(repaired, None)

#         try:
#             return self._wrap_for_ragas(None, json.dumps(model_response))
#         except Exception:
#             s = str(model_response)
#             self._log_raw_output(s)
#             return self._wrap_for_ragas(None, s)


# # -----------------------
# # Pipeline utilities
# # -----------------------

# def run_pipeline(question, agent, user_roles, headings):
#     logger.debug("Running pipeline for question: %s", question)
#     queries = agent.generate_queries(question)
#     retriever = Embeddings()
#     retrieved_docs = retriever.retrieve_documents(queries, user_roles, headings)

#     fusion = RankFusion()
#     fused_docs = fusion.reciprocal_rank_fusion(retrieved_docs)

#     final_answer = agent.generate_answer(question, fused_docs)

#     logger.debug("Pipeline complete for question: %s", question)
#     return {
#         "user_input": question,
#         "retrieved_contexts": fused_docs,
#         "response": final_answer,
#     }


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
#     parser.add_argument(
#         "--questions",
#         required=True,
#         help="Path to a text file with one question per line.",
#     )
#     args = parser.parse_args()

#     logger.info("Starting ragas evaluation script")
#     logger.debug("Parsed args: %s", args)

#     user_roles = ["engineer"]
#     headings = ["default", "Confidential"]

#     model_id = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-7b-instruct-v0:2")
#     aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
#     aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
#     region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

#     session = boto3.Session(
#         aws_access_key_id=aws_access_key_id,
#         aws_secret_access_key=aws_secret_access_key,
#         region_name=region_name,
#     )

#     bedrock_runtime_client = session.client("bedrock-runtime")
#     langchain_llm = BedrockSyncLLM(client=bedrock_runtime_client, model_id=model_id)

#     embedding_model = BedrockEmbeddings(
#         client=bedrock_runtime_client,
#         model_id="amazon.titan-embed-text-v2:0",
#     )

#     my_bedrock_client = BedrockClient()
#     agent = RAGAgent(my_bedrock_client, model_id)

#     with open(args.questions, "r", encoding="utf-8") as f:
#         questions = [line.strip() for line in f if line.strip()]

#     if not questions:
#         logger.warning("No questions found in the provided file. Exiting.")
#         return

#     eval_data = [run_pipeline(q, agent, user_roles, headings) for q in questions]
#     dataset = EvaluationDataset.from_list(eval_data)

#     results = evaluate(
#         llm=langchain_llm,
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy],
#         embeddings=embedding_model,
#     )

#     print(results)
#     results.to_csv("ragas_results.csv", index=False)
#     logger.info("Saved results to ragas_results.csv")


##########################################
### TRULENS
##########################################


# import os, argparse, logging, time, json
# import boto3
# import numpy as np
# from dataclasses import dataclass, field
# from typing import List
# from dotenv import load_dotenv

# # === logging ===
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # === TruLens imports ===
# # IMPORTANT: enable tracing BEFORE importing TruLens instrumentation
# load_dotenv()
# os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

# from trulens.core import Feedback, TruSession
# from trulens.core.otel.instrument import instrument
# from trulens.otel.semconv.trace import SpanAttributes
# from trulens.apps.app import TruApp
# from trulens.providers.litellm import LiteLLM
# from trulens.dashboard import run_dashboard  # optional

# # === Your app pieces ===
# from ..bedrock_client import BedrockClient
# from ..agent import RAGAgent
# from ..embeddings import Embeddings
# from ..rank_fusion import RankFusion
# import re

# # from langchain_aws import BedrockEmbeddings  # not used here


# # -------------------------
# # Helpers
# # -------------------------
# def _normalize_context_texts(docs) -> List[str]:
#     """
#     Convert a heterogeneous docs structure into a list[str] passages.
#     Handles common shapes like LangChain Documents, dicts, tuples, or raw strings.
#     Falls back to repr if nothing else fits.
#     """
#     texts: List[str] = []
#     if docs is None:
#         return texts

#     for d in docs:
#         # LangChain Document-like
#         if hasattr(d, "page_content"):
#             texts.append(str(getattr(d, "page_content")))
#             continue

#         # dict-like common keys
#         if isinstance(d, dict):
#             for k in ("page_content", "content", "text", "body", "chunk"):
#                 if k in d and isinstance(d[k], (str, bytes)):
#                     texts.append(d[k].decode("utf-8") if isinstance(d[k], bytes) else str(d[k]))
#                     break
#             else:
#                 texts.append(repr(d))
#             continue

#         # tuple/list like (text, score) etc.
#         if isinstance(d, (list, tuple)) and len(d) > 0:
#             candidate = d[0]
#             if isinstance(candidate, (str, bytes)):
#                 texts.append(candidate.decode("utf-8") if isinstance(candidate, bytes) else candidate)
#             elif hasattr(candidate, "page_content"):
#                 texts.append(str(candidate.page_content))
#             else:
#                 texts.append(repr(d))
#             continue

#         # raw strings
#         if isinstance(d, (str, bytes)):
#             texts.append(d.decode("utf-8") if isinstance(d, bytes) else d)
#             continue

#         # fallback
#         texts.append(repr(d))

#     # optional: trim whitespace and drop empties
#     texts = [t.strip() for t in texts if t and t.strip()]
#     return texts


# # -------------------------
# # Wrap your pipeline as an App TruLens can record
# # -------------------------
# @dataclass
# class RAGPipelineApp:
#     agent: object
#     retriever: object
#     fusion: object
#     last_contexts: List[str] = field(default_factory=list)  # avoid a mutable None default

#     @instrument(
#         span_type=SpanAttributes.SpanType.RETRIEVAL,
#         attributes={
#             SpanAttributes.RETRIEVAL.QUERY_TEXT: "question",
#             # ⛔️ SpanAttributes.RETRIEVAL.CONTEXTS  # not in your version
#             "ai.observability.retrieval.contexts": "return",  # ✅ works across versions
#         },
#     )
#     def retrieve(self, question: str, user_roles, headings):
#         queries = self.agent.generate_queries(question)
#         retrieved_docs = self.retriever.retrieve_documents(queries, user_roles, headings)

#         # You can keep your fusion on objects:
#         fused_docs = self.fusion.reciprocal_rank_fusion(retrieved_docs)

#         # But TruLens needs plain strings for contexts:
#         contexts = _normalize_context_texts(fused_docs)
#         self.last_contexts = contexts

#         # Return list[str] so instrument() places it into RETRIEVAL.CONTEXTS
#         return contexts

#     @instrument(
#         span_type=SpanAttributes.SpanType.GENERATION,
#     )
#     def generate(self, question: str, contexts: List[str]):
#         # Pass plain-text contexts to your agent
#         return self.agent.generate_answer(question, contexts)

#     @instrument(
#         span_type=SpanAttributes.SpanType.RECORD_ROOT,
#         attributes={
#             SpanAttributes.RECORD_ROOT.INPUT: "question",
#             # Ensure the return is a STRING to avoid OUTPUT type warnings.
#             SpanAttributes.RECORD_ROOT.OUTPUT: "return",
#         },
#     )
#     def query(self, question: str, user_roles, headings):
#         contexts = self.retrieve(question, user_roles, headings)
#         answer = self.generate(question, contexts)
#         return answer  # string only


# def pick_accessible_model(session, preferred_id: str, region_name: str) -> str:
#     bedrock = session.client("bedrock", region_name=region_name)
#     resp = bedrock.list_foundation_models(
#         byOutputModality="TEXT",
#         byInferenceType="ON_DEMAND"
#     )
#     available_ids = {m["modelId"] for m in resp.get("modelSummaries", [])}


#     if preferred_id in available_ids:
#         return preferred_id

#     preferred_candidates = [
#         mid for mid in available_ids
#         if any(s in mid.lower() for s in ("instruct", "chat", "claude", "llama", "mistral", "nova", "qwen"))
#     ]
#     if preferred_candidates:
#         return sorted(preferred_candidates)[0]

#     if available_ids:
#         return sorted(available_ids)[0]

#     raise RuntimeError(
#         "No accessible Bedrock TEXT models found in this region/account. "
#         "Enable at least one in the Bedrock console."
#     )

# def _extract_rating_0_3(text: str) -> int:
#     # Try JSON first
#     try:
#         obj = json.loads(text)
#         if isinstance(obj, dict):
#             val = obj.get("rating", obj.get("score", None))
#             if val is not None:
#                 val = int(val)
#                 if val in (0,1,2,3):
#                     return val
#     except Exception:
#         pass

#     # Fallback: keyed regex to avoid picking up digits from "NIS2", "27001", etc.
#     m = re.search(r'"(?:rating|score)"\s*:\s*([0-3])\b', text)
#     if m:
#         return int(m.group(1))

#     # As last resort, look for a standalone 0–3 preceded by "Score" or "Rating"
#     m = re.search(r'(?:Score|Rating)\D*([0-3])\b', text, re.IGNORECASE)
#     if m:
#         return int(m.group(1))

#     raise ValueError("Could not extract rating in [0–3].")


# def judge_groundedness(contexts: List[str], output: str) -> int:
#     raw = judge.groundedness_measure_with_cot_reasons(context=contexts, output=output)
#     breakpoint()
#     return _extract_rating_0_3(raw)

# def judge_answer_relevance(question: str, output: str) -> int:
#     raw = judge.relevance_with_cot_reasons(input=question, output=output)
#     return _extract_rating_0_3(raw)

# def judge_context_relevance(question: str, context_chunk: str) -> int:
#     raw = judge.context_relevance_with_cot_reasons(input=question, context=context_chunk)
#     return _extract_rating_0_3(raw)

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with TruLens")
#     parser.add_argument("--questions", required=True, help="Path to a text file with one question per line.")
#     parser.add_argument("--dashboard", action="store_true", help="Launch TruLens dashboard locally.")
#     args = parser.parse_args()

#     logger.info("Starting TruLens evaluation script")

#     # --- Your app config ---
#     user_roles = ["engineer"]
#     headings = ["default", "Confidential"]

#     aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
#     aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
#     region_name = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

#     session = boto3.Session(
#         aws_access_key_id=aws_access_key_id,
#         aws_secret_access_key=aws_secret_access_key,
#         region_name=region_name,
#     )
#     bedrock_runtime_client = session.client("bedrock-runtime")
    
#     model_id = pick_accessible_model(session, preferred_id="qwen.qwen3-235b-a22b-2507-v1:0", region_name=region_name) 


#     # Choose a fallback model if none specified or not enabled
#     if not model_id:
#         model_id = pick_accessible_model(session, preferred_id="eu.amazon.nova-lite-v1:0", region_name=region_name)
    

#     my_bedrock_client = BedrockClient()
#     agent = RAGAgent(my_bedrock_client, model_id)
#     retriever = Embeddings()
#     fusion = RankFusion()

#     with open(args.questions, "r", encoding="utf-8") as f:
#         questions = [line.strip() for line in f if line.strip()]

#     if not questions:
#         logger.warning("No questions found in the provided file. Exiting.")
#         return

#     # -------------------------
#     # TruLens setup (Bedrock as judge)
#     # -------------------------
#     tru = TruSession(
#         database_url="sqlite:///trulens.sqlite",
#         database_redact_keys=True
#     )

#     # Make sure the LiteLLM engine string matches your Bedrock model id style.
#     # Use a general-purpose, enabled text model as judge:
#     judge_id = pick_accessible_model(session, preferred_id="qwen.qwen3-235b-a22b-2507-v1:0", region_name=region_name) 
#     judge = LiteLLM(
#         model_engine=f"bedrock/{judge_id}",
#         completion_args={
#             "aws_region_name": region_name,
#             "temperature": 0,
#             "max_tokens": 512,     # avoid truncation before “**Score:** N”
#             "timeout": 60,         # optional
#             "max_retries": 3,       # optional
#             "response_format": {"type": "json_object"}
#         }
#     )

#     #breakpoint()

#     # Define feedbacks (RAG Triad)
#     f_groundedness = (
#         Feedback(judge_groundedness, name="Groundedness")
#         .on_context(collect_list=True)  # use all context chunks
#         .on_output()
#     )

#     f_answer_rel = (
#         Feedback(judge.relevance_with_cot_reasons, name="Answer Relevance")
#         .on_input()   # the question
#         .on_output()  # the answer
#     )

#     f_ctx_rel = (
#         Feedback(judge.context_relevance_with_cot_reasons, name="Context Relevance")
#         .on_input()                      # the question
#         .on_context(collect_list=False)  # score each chunk individually
#         .aggregate(np.mean)              # aggregate to a single score per request
#     )

#     # -------------------------
#     # Wrap pipeline as TruLens app
#     # -------------------------
#     app = RAGPipelineApp(agent=agent, retriever=retriever, fusion=fusion)
#     tru_app = TruApp(
#         app,
#         app_name="RAGPipeline",
#         app_version=model_id or "unknown-model",
#         feedbacks=[f_groundedness, f_answer_rel, f_ctx_rel],
#         async_feedback=False
#     )

#     # -------------------------
#     # Run evaluation over questions and record results
#     # -------------------------
#     all_outputs = []
#     with tru_app as _:
#         for q in questions:
#             answer = app.query(q, user_roles, headings)  # returns STRING now
#             # tiny delay to give async feedback threads space (optional)
#             time.sleep(0.2)
#             all_outputs.append({
#                 "user_input": q,
#                 "retrieved_contexts": app.last_contexts or [],
#                 "response": answer
#             })

#     # Persist
#     time.sleep(2)
#     df, _schema = tru.get_records_and_feedback()
#     df.to_csv("trulens_results.csv", index=False)
#     logger.info("Saved TruLens results to trulens_results.csv")

#     with open("pipeline_outputs.json", "w", encoding="utf-8") as f:
#         json.dump(all_outputs, f, ensure_ascii=False, indent=2)

#     if args.dashboard:
#         run_dashboard(tru)


# if __name__ == "__main__":
#     main()

# --- lenient wrapper for DeepEval to survive sloppy JSON judges ---
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm


class LenientLiteLLMModel(DeepEvalBaseLLM):
    """
    LiteLLM (Bedrock) -> DeepEval bridge with robust JSON coercion:
      - Supports DeepEval schemas for Statements, Claims, Truths, Verdicts
      - Handles pydantic v1/v2
      - Returns only the parsed object (NOT a tuple)
    """

    def __init__(self, *, model: str, aws_region_name: str, **kwargs):
        self.model = model
        self.aws_region_name = aws_region_name
        self.kwargs = kwargs
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
                            out.append(v[k]); break
        elif isinstance(values, str):
            out = [values]
        if not out:
            out = [""]
        return out

    # ---- verdicts coercion variants ----
    def _coerce_verdicts_obj(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preferred for many DeepEval builds: verdicts: List[{verdict: Literal['yes','no','idk']}]
        """
        src = raw.get("verdicts")
        out = []
        if isinstance(src, list):
            for v in src:
                if isinstance(v, dict) and "verdict" in v:
                    val = str(v["verdict"]).lower()
                    if val not in ("yes", "no", "idk"):
                        val = "idk"
                    out.append({"verdict": val})
                elif isinstance(v, str):
                    val = v.lower()
                    if val not in ("yes", "no", "idk"):
                        val = "idk"
                    out.append({"verdict": val})
        elif isinstance(raw.get("verdict"), str):
            val = raw["verdict"].lower()
            if val not in ("yes", "no", "idk"):
                val = "idk"
            out = [{"verdict": val}]
        if not out:
            out = [{"verdict": "no"}]
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
            stmts = raw.get("statements") or raw.get("claims") or raw.get("truths") or raw.get("items") or raw.get("text")
            return {"statements": self._wrap_plain_list_of_str(stmts)}

        if "truths" in fields:
            truths = raw.get("truths") or raw.get("claims") or raw.get("statements") or raw.get("items") or raw.get("text")
            return {"truths": self._wrap_plain_list_of_str(truths)}

        if "claims" in fields:
            claims = raw.get("claims") or raw.get("truths") or raw.get("statements") or raw.get("items") or raw.get("text")
            return {"claims": self._wrap_plain_list_of_str(claims)}

        if "verdicts" in fields:
            return self._coerce_verdicts_obj(raw)

        return raw or {}

    def _coerce_legacy(self, raw: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Legacy coercion: fallback shapes used by some older DeepEval builds.
        - truths/claims as list[dict]
        - verdicts as list[str]
        """
        fields = self._pd_fields(schema)

        if "truths" in fields:
            truths = raw.get("truths") or raw.get("claims") or raw.get("statements") or raw.get("items") or raw.get("text")
            return {"truths": [{"truth": t} for t in self._wrap_plain_list_of_str(truths)]}

        if "claims" in fields:
            claims = raw.get("claims") or raw.get("truths") or raw.get("statements") or raw.get("items") or raw.get("text")
            return {"claims": [{"claim": c} for c in self._wrap_plain_list_of_str(claims)]}

        if "verdicts" in fields:
            return self._coerce_verdicts_str(raw)

        if "statements" in fields:
            stmts = raw.get("statements") or raw.get("claims") or raw.get("truths") or raw.get("items") or raw.get("text")
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

    # ---- DeepEval LLM interface ----
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        res = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            aws_region_name=self.aws_region_name,
            **self.kwargs,
        )
        choice = res.choices[0]
        text = getattr(choice, "message", {}).get("content") if hasattr(choice, "message") else getattr(choice, "text", "")
        return self._parse_with_schema(text or "", schema)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        res = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            aws_region_name=self.aws_region_name,
            **self.kwargs,
        )
        choice = res.choices[0]
        text = getattr(choice, "message", {}).get("content") if hasattr(choice, "message") else getattr(choice, "text", "")
        return self._parse_with_schema(text or "", schema)



import os, argparse, logging, time, json
import boto3
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

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
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import evaluate
from deepeval.models import LiteLLMModel

# -------------------------
# Helpers
# -------------------------
def _normalize_context_texts(docs) -> List[str]:
    texts: List[str] = []
    if docs is None:
        return texts
    for d in docs:
        if hasattr(d, "page_content"):
            texts.append(str(getattr(d, "page_content"))); continue
        if isinstance(d, dict):
            for k in ("page_content", "content", "text", "body", "chunk"):
                if k in d and isinstance(d[k], (str, bytes)):
                    texts.append(d[k].decode("utf-8") if isinstance(d[k], bytes) else str(d[k])); break
            else:
                texts.append(repr(d)); continue
            continue
        if isinstance(d, (list, tuple)) and len(d) > 0:
            candidate = d[0]
            if isinstance(candidate, (str, bytes)):
                texts.append(candidate.decode("utf-8") if isinstance(candidate, bytes) else candidate)
            elif hasattr(candidate, "page_content"):
                texts.append(str(candidate.page_content))
            else:
                texts.append(repr(d))
            continue
        if isinstance(d, (str, bytes)):
            texts.append(d.decode("utf-8") if isinstance(d, bytes) else d); continue
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
    resp = bedrock.list_foundation_models(byOutputModality="TEXT", byInferenceType="ON_DEMAND")
    ids = {m["modelId"] for m in resp.get("modelSummaries", [])}
    if preferred_id in ids: return preferred_id
    cands = [mid for mid in ids if any(s in mid.lower() for s in ("instruct","chat","claude","llama","mistral","nova","qwen"))]
    if cands: return sorted(cands)[0]
    if ids: return sorted(ids)[0]
    raise RuntimeError("No accessible Bedrock TEXT models found.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with DeepEval (reference-free metrics)")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--csv", default="deepeval_results.csv")
    parser.add_argument("--json", default="pipeline_outputs.json")
    args = parser.parse_args()

    logger.info("Starting DeepEval evaluation (reference-free: faithfulness + answer relevancy)")

    user_roles = ["engineer"]
    headings = ["default", "Confidential"]

    region_name = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )
    _ = session.client("bedrock-runtime")

    model_id = pick_accessible_model(session, preferred_id="qwen.qwen3-235b-a22b-2507-v1:0", region_name=region_name)
    if not model_id:
        model_id = pick_accessible_model(session, preferred_id="eu.amazon.nova-lite-v1:0", region_name=region_name)

    my_bedrock_client = BedrockClient()
    agent = RAGAgent(my_bedrock_client, model_id)
    retriever = Embeddings()
    fusion = RankFusion()
    app = RAGPipelineApp(agent=agent, retriever=retriever, fusion=fusion)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    if not questions:
        logger.warning("No questions found in the provided file. Exiting.")
        return

    # LiteLLM bridge to Bedrock (no structured JSON required)
    judge_model = LenientLiteLLMModel(
        model=f"bedrock/{model_id}",   # or a separate judge_id like a Nova model
        aws_region_name=region_name,
        temperature=0,
        max_tokens=512,
        timeout=60,
        # Hint JSON mode if supported; harmless otherwise:
        response_format={"type": "json_object"},
    )

    faithfulness = FaithfulnessMetric(model=judge_model, threshold=0.0)
    answer_rel = AnswerRelevancyMetric(model=judge_model, threshold=0.0)

    all_outputs = []
    test_cases: List[LLMTestCase] = []
    for q in questions:
        answer = app.query(q, user_roles, headings)
        contexts = app.last_contexts or []
        all_outputs.append({"user_input": q, "retrieved_contexts": contexts, "response": answer})
        test_cases.append(LLMTestCase(input=q, actual_output=answer, retrieval_context=contexts))
        time.sleep(0.05)

    # Run eval (no max_concurrency arg for older DeepEval versions)
    metrics = [faithfulness, answer_rel]
    evaluate(test_cases=test_cases, metrics=metrics)

    # Persist per-question scores
    rows = []
    for case in test_cases:
        f = FaithfulnessMetric(model=judge_model, threshold=0.0)
        ar = AnswerRelevancyMetric(model=judge_model, threshold=0.0)
        f.measure(case); ar.measure(case)
        rows.append({
            "question": case.input,
            "faithfulness_score": getattr(f, "score", None),
            "answer_relevancy_score": getattr(ar, "score", None),
            "num_context_chunks": len(case.retrieval_context or []),
        })

    import csv
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "faithfulness_score", "answer_relevancy_score", "num_context_chunks"
        ])
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Saved DeepEval metric results to {args.csv}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved pipeline outputs to {args.json}")

if __name__ == "__main__":
    # Optional noise control:
    # os.environ["CHROMA_TELEMETRY_DISABLED"] = "TRUE"
    # os.environ["LITELLM_LOG"] = "ERROR"
    main()

