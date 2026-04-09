import os
import json
import logging
import random
import time
from typing import Dict, Any, List

import boto3
import botocore
from botocore.config import Config

logger = logging.getLogger(__name__)

# Non-retryable error codes — fail fast instead of wasting retries
_NON_RETRYABLE_CODES = frozenset({
    "AccessDeniedException",
    "UnrecognizedClientException",
    "ValidationException",
    "ResourceNotFoundException",
    "ModelNotReadyException",
})


class BedrockClient:
    """
    Simple wrapper around the AWS Bedrock Runtime API that
    picks up creds and region from environment variables.
    """

    def __init__(self):
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        if not aws_access_key_id or not aws_secret_access_key:
            raise EnvironmentError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set. "
                "See .env.example for the required environment variables."
            )

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self.client = session.client(
            'bedrock-runtime',
            config=Config(
                retries={"max_attempts": 10, "mode": "standard"},
                read_timeout=300,
                connect_timeout=10,
            )
        )

    def get_account_id(self) -> str:
        sts = boto3.Session().client('sts')
        return sts.get_caller_identity()['Account']

    def list_models(self) -> List[Dict[str, Any]]:
        meta = self.client.meta.client('bedrock')
        resp = meta.list_foundation_models()
        return resp.get('foundationModels', [])

    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        if isinstance(prompt, list):
            try:
                input_text = "\n".join([msg.content for msg in prompt])
            except AttributeError:
                raise TypeError("Prompt list must contain objects with a 'content' attribute.")
        elif hasattr(prompt, "content"):  # Ein einzelnes HumanMessage-Objekt
            input_text = prompt.content
        elif isinstance(prompt, str):
            input_text = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        payload_messages = [
            {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
            {"role": "user",   "content": [{"text": input_text}]}
        ]

        payload = {
            "messages": payload_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }

        inference_cfg = payload["inferenceConfig"]
        max_retries = 8
        base = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                resp = self.client.converse(
                    modelId=model_id,
                    system=[{"text": "You are a helpful assistant."}],
                    messages=[
                        {"role": "user", "content": [{"text": input_text}]}
                    ],
                    inferenceConfig=inference_cfg
                )
                return resp
            except botocore.exceptions.ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                # Fail fast on non-retryable errors (auth, validation, etc.)
                if code in _NON_RETRYABLE_CODES:
                    raise
                if code in (
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "RequestLimitExceeded",
                    "ServiceQuotaExceededException",
                ):
                    sleep = min(8.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                    logger.warning(
                        "Bedrock throttled (attempt %d/%d, code=%s). "
                        "Retrying in %.1fs…", attempt + 1, max_retries, code, sleep,
                    )
                    time.sleep(sleep)
                    continue
                raise

        raise RuntimeError(
            f"Bedrock request failed after {max_retries} retries (last error: throttling)"
        )
