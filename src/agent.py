import os
from typing import List

from .bedrock_client import BedrockClient
from .retriever import VectorRetriever
from langchain.prompts import ChatPromptTemplate
import json
# from .prompt_templates import RAG_PROMPT

class RAGAgent:
    """
    Retrieval-Augmented Generation agent using AWS Bedrock as the LLM backend.
    """
    def __init__(
        self,
        #retriever: VectorRetriever,
        bedrock_client: BedrockClient,
        model_id: str,
        #prompt_template: str = RAG_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        k: int = 5,
    ):
       # self.retriever = retriever
        self.bedrock = bedrock_client
        self.model_id = model_id
        #self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.k = k

    def generate_queries(self, question) -> str:
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant. Generate five different rewrites of the user's question to help retrieve relevant documents from a vector database. Separate each rewritten question with a newline. Only use newlines after the first four questions. Just create the questions without any additional text.

        User's question: {question}
        """)
        return self.answer_question(prompt.format_messages(question=question))

    def generate_answer(self, question, context_docs):
        context_texts = [doc[0] if isinstance(doc, tuple) else doc for doc in context_docs[:5]]
        context = "\n\n".join(context_texts)  # Limit context to top 5 documents
        answer_prompt = f"""
            Context:
            {context}

            Question: {question}

            Answer:"""
        return self.answer_question(answer_prompt)

    def answer_from_db(self, question: str) -> str:
        return self.bedrock.retrieve_from_db(self.model_id, question)
    def answer_question(self, question: str) -> str:
        """
        Retrieves relevant document chunks and queries Bedrock to generate an answer.
        """
        # 1. Retrieve top-k relevant chunks
        #docs: List[str] = self.retriever.retrieve(question, k=self.k)

        # # 2. Build the prompt
        # context = "\n\n".join(docs)
        # prompt = self.prompt_template.format(context=context, question=question)

        # 3. Invoke the Bedrock model
        response = self.bedrock.invoke_model(
            model_id=self.model_id,
            prompt=question,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response['output']['message']['content'][0]['text']
        
        #         # 4. Parse and return the generated answer
        # # Assuming response['results'] is a list of dicts with 'content'
        # results = response.get('results') or []
        # if results and isinstance(results, list):
        #     # Join multiple generations if present
        #     return "\n".join(res.get('content', '') for res in results)
        # # Fallback: return full response JSON as string
        # return str(response)

    def judge_question_domain(self, question: str, *, min_score: float = 0.60):
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

        prompt = f"""
        You are a strict domain gatekeeper. Decide whether the USER QUESTION is IN-DOMAIN for the DOMAIN.

        Return ONLY a compact JSON object without any markdown or such wrapping it, starting with curly braces, with keys:
        - "in_domain": true or false
        - "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
        - "rationale": a short one-sentence explanation

        DOMAIN:
        {domain_desc}

        USER QUESTION:
        {question}

        JSON:
        """

        response = self.bedrock.invoke_model(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )
        raw = response['output']['message']['content'][0]['text'].strip()

        try:
            data = json.loads(raw)
            in_domain = bool(data.get("in_domain"))
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", "")).strip()
        except Exception:
            # If parsing fails, treat as out-of-domain
            in_domain, score, rationale = False, 0.0, "Judge JSON parse failed."

        # Apply threshold
        in_domain = in_domain and (score >= min_score)
        return in_domain, score, rationale
