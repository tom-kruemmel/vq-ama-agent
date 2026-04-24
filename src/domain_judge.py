import json
import logging

from .bedrock_client import BedrockClient
from .prompt_templates import JUDGE_DOMAIN_TWO_STAGE

logger = logging.getLogger(__name__)

# Best-performing judge model based on evaluation (F1 0.857 in-domain / 0.870 out-of-domain)
DEFAULT_JUDGE_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

# Default domain description for virtualQ
_DEFAULT_DOMAIN_DESC = (
    "Questions specifically about the software as a service company virtualQ providing callback solutions for call centers and the company's technology stack. "
    "This includes: virtualQ's products, APIs/SDKs, architecture, cloud providers, "
    "datastores, infrastructure, integrations, deployment/CICD, observability, "
    "security/compliance, and engineering practices at virtualQ. Be sure to include "
    "technology related questions in the domain especially if they are related to "
    "telephony. Include all questions that are reasonable to ask a technology company. "
    "It excludes unrelated general knowledge and questions about other companies."
)


class DomainJudge:
    """Decides whether a user question is in-domain using an LLM."""

    def __init__(
        self,
        bedrock_client: BedrockClient,
        model_id: str = DEFAULT_JUDGE_MODEL_ID,
        prompt_template: str = JUDGE_DOMAIN_TWO_STAGE,
        domain_desc: str = _DEFAULT_DOMAIN_DESC,
    ):
        self.bedrock = bedrock_client
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.domain_desc = domain_desc

    def judge(
        self,
        question: str,
        *,
        chat_history: str = "",
        min_score: float = 0.60,
    ) -> tuple[bool, float, str]:
        """Return *(in_domain, score, rationale)* for *question*."""
        prompt = self.prompt_template.format(
            domain_desc=self.domain_desc,
            question=question,
            chat_history=chat_history,
        )

        response = self.bedrock.invoke_model(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
        )

        raw = None
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                raw = block["text"].strip()
                break
        if raw is None:
            raw = str(response["output"]["message"]["content"])

        try:
            data = json.loads(raw)
            in_domain = bool(data.get("in_domain"))
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Judge JSON parse failed: %s — raw: %.200s", exc, raw)
            in_domain, score, rationale = False, 0.0, "Judge JSON parse failed."

        in_domain = in_domain and (score >= min_score)
        return in_domain, score, rationale
