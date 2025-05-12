import os
from typing import List

from .bedrock_client import BedrockClient
from .retriever import VectorRetriever
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
        return response["results"][0]["outputText"]
        
        #         # 4. Parse and return the generated answer
        # # Assuming response['results'] is a list of dicts with 'content'
        # results = response.get('results') or []
        # if results and isinstance(results, list):
        #     # Join multiple generations if present
        #     return "\n".join(res.get('content', '') for res in results)
        # # Fallback: return full response JSON as string
        # return str(response)
