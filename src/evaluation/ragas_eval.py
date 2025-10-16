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


# if __name__ == "__main__":
#     main()
import os, argparse, logging
import boto3
import numpy as np
from dataclasses import dataclass

# === your imports (unchanged) ===
# from your_module import Embeddings, RankFusion, BedrockClient, RAGAgent, BedrockSyncLLM, BedrockEmbeddings
# from ragas import ...   # <-- removed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === TruLens imports ===
from trulens.core import Feedback, TruSession
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.providers.bedrock import Bedrock
from trulens.providers.litellm import LiteLLM
from trulens.dashboard import run_dashboard  # optional
from ..bedrock_client import BedrockClient
from ..agent import RAGAgent
from ..embeddings import Embeddings
from ..rank_fusion import RankFusion
from langchain_aws import BedrockEmbeddings
import boto3, botocore, os

# -------------------------
# Wrap your pipeline as an App TruLens can record
# -------------------------
# --- add at VERY top of file, before any trulens imports ---
# ensure this is set BEFORE any trulens imports
import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

from dataclasses import dataclass
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
import time
@dataclass
class RAGPipelineApp:
    agent: object
    retriever: object
    fusion: object

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "question",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, question: str, user_roles, headings):
        queries = self.agent.generate_queries(question)
        retrieved_docs = self.retriever.retrieve_documents(queries, user_roles, headings)
        fused_docs = self.fusion.reciprocal_rank_fusion(retrieved_docs)
        return fused_docs

    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        # GENERATION has no standard attributes — omit semconv attrs here.
        # If you want custom attrs, you can do: attributes={"gen_input": "question", "gen_output": "return"}
    )
    def generate(self, question: str, fused_docs):
        return self.agent.generate_answer(question, fused_docs)

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "question",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, question: str, user_roles, headings):
        fused = self.retrieve(question, user_roles, headings)
        answer = self.generate(question, fused)
        return {
            "user_input": question,
            "retrieved_contexts": fused,
            "response": answer,
        }


def pick_accessible_model(session, preferred_id: str, region_name: str) -> str:
    """
    Return `preferred_id` if accessible in this account/region; otherwise
    return the first accessible TEXT or TEXT_GENERATION model.
    """
    bedrock = session.client("bedrock", region_name=region_name)
    # Limit to text-capable on-demand models
    resp = bedrock.list_foundation_models(
        byOutputModality="TEXT",
        byInferenceType="ON_DEMAND"
    )
    available_ids = {m["modelId"] for m in resp.get("modelSummaries", [])}

    if preferred_id in available_ids:
        return preferred_id

    # Prefer instruction/chat tuned models if present
    preferred_candidates = [
        mid for mid in available_ids
        if any(s in mid.lower() for s in ("instruct", "chat", "claude", "llama", "mistral"))
    ]
    if preferred_candidates:
        return sorted(preferred_candidates)[0]

    # As last resort, pick any TEXT model
    if available_ids:
        return sorted(available_ids)[0]

    raise RuntimeError(
        "No accessible Bedrock TEXT models found in this region/account. "
        "Enable at least one in the Bedrock console."
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with TruLens")
    parser.add_argument("--questions", required=True, help="Path to a text file with one question per line.")
    parser.add_argument("--dashboard", action="store_true", help="Launch TruLens dashboard locally.")
    args = parser.parse_args()

    logger.info("Starting TruLens evaluation script")
    logger.debug("Parsed args: %s", args)

    # --- Your app config (unchanged) ---
    user_roles = ["engineer"]
    headings = ["default", "Confidential"]

    model_id = os.getenv("BEDROCK_MODEL_ID")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    bedrock_runtime_client = session.client("bedrock-runtime")

    rt = boto3.client("bedrock-runtime", region_name=region_name)
    test_model = "eu.amazon.nova-lite-v1:0"  # pick a TEXT model you’ve enabled in eu-central-1
    # try:
    #     rt.invoke_model(
    #         modelId=test_model,
    #         body=b'{"prompt":"ping","max_gen_len":1,"temperature":0}'
    #     )
    #     print("✅ Bedrock invoke works with", test_model, "in", region_name)
    # except botocore.exceptions.ClientError as e:
    #     raise SystemExit(f"❌ Bedrock invoke failed: {e}")

    # If you still need these elsewhere in your stack:
    # langchain_llm = BedrockSyncLLM(client=bedrock_runtime_client, model_id=model_id)
    # embedding_model = BedrockEmbeddings(client=bedrock_runtime_client, model_id="amazon.titan-embed-text-v2:0")

    my_bedrock_client = BedrockClient()
    agent = RAGAgent(my_bedrock_client, model_id)
    retriever = Embeddings()
    fusion = RankFusion()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        logger.warning("No questions found in the provided file. Exiting.")
        return

    # -------------------------
    # TruLens setup (Bedrock as judge)
    # -------------------------
    tru = TruSession(
        database_url="sqlite:///trulens.sqlite",
        database_redact_keys=True
    )
    #tru.reset_database()    
    # if hasattr(tru, "migrate_database"):
    #     tru.migrate_database()
    # else:
    #     # If migrate isn't available (older version), reset will recreate tables (wipes data).
    #     tru.reset_database()
    # Use a lightweight Bedrock model as the evaluator ("judge").
    # You can use the same region as your runtime client.
    # Pick any Bedrock chat model that suits evaluation cost/speed/quality tradeoffs.
    judge = LiteLLM(
        model_engine="bedrock/eu.mistral.pixtral-large-2502-v1:0",
        completion_args={"aws_region_name": "eu-central-1"}
    )
    # Docs: Bedrock provider & quickstart recipes. :contentReference[oaicite:1]{index=1}

    # Define the RAG Triad feedback functions.
    # Groundedness: Is the answer supported by the retrieved context?
    f_groundedness = (
        Feedback(judge.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on_context(collect_list=True)  # use all context chunks
        .on_output()
    )

    # Answer Relevance: Does the answer address the user's question?
    f_answer_rel = (
        Feedback(judge.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()   # the question
        .on_output()  # the answer
    )

    # Context Relevance: Are the retrieved chunks relevant to the question?
    f_ctx_rel = (
        Feedback(judge.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()                    # the question
        .on_context(collect_list=False)  # score each chunk individually
        .aggregate(np.mean)            # aggregate to a single score per request
    )

    # -------------------------
    # Wrap your pipeline as a TruLens app
    # -------------------------
    app = RAGPipelineApp(agent=agent, retriever=retriever, fusion=fusion)
    tru_app = TruApp(
        app,
        app_name="RAGPipeline",
        app_version=model_id,
        feedbacks=[f_groundedness, f_answer_rel, f_ctx_rel],
    )
    # Pattern follows the TruLens quickstart & Bedrock cookbook. :contentReference[oaicite:2]{index=2}

    # -------------------------
    # Run evaluation over questions and record results
    # -------------------------
    all_outputs = []
    with tru_app as recording:
        for q in questions:
            result = app.query(q, user_roles, headings)
            time.sleep(1)
            all_outputs.append(result)

    # Pull records + feedback into a dataframe and save.
    df = tru.get_records_and_feedback()[0]  # returns (df, schema) in recent versions
    df.to_csv("trulens_results.csv", index=False)
    logger.info("Saved TruLens results to trulens_results.csv")

    # Optional: also write your raw pipeline outputs for convenience.
    import json
    with open("pipeline_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    # Optional dashboard to inspect traces & scores interactively:
    if args.dashboard:
        run_dashboard(tru)

if __name__ == "__main__":
    main()
