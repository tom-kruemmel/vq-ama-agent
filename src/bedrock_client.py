import os
import json
import boto3
from typing import Dict, Any, List
from langchain_community.chat_models.bedrock import BedrockChat
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings

# 🔽 minimal additions for throttling handling
import time, random
import botocore
from botocore.config import Config
# 🔼

class BedrockClient:
    """
    Simple wrapper around the AWS Bedrock Runtime API that
    picks up creds and region from environment variables.
    """

    def __init__(self):
        # Read credentials + region from env
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name           = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        # Create a session that will automatically sign requests
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        # Bedrock uses the 'bedrock-runtime' client for inference
        # 🔽 minimal change: add retry-friendly client config
        self.client = session.client(
            'bedrock-runtime',
            config=Config(
                retries={"max_attempts": 10, "mode": "standard"},
                read_timeout=60,
                connect_timeout=10,
            )
        )
        # 🔼

    
    def get_account_id(self) -> str:
        sts = boto3.Session().client('sts')
        return sts.get_caller_identity()['Account']

    def list_models(self) -> List[Dict[str, Any]]:
        # Some regions require the 'bedrock' meta-client
        meta = self.client.meta.client('bedrock')
        resp = meta.list_foundation_models()
        return resp.get('foundationModels', [])

    # def create_inference_profile(
    #     self):
    #     control_plane = boto3.client('bedrock', region_name='eu-central-1')
    #     res = control_plane.create_inference_profile(
    #         inferenceProfileName='my-pixtral-profile',
    #         modelSource={
    #             'copyFrom': 'arn:aws:bedrock:eu-central-1::foundation-model/mistral.pixtral-large-2502-v1:0'
    #         },
    #         description='Profile for Pixtral Large',
    #         tags=[{'key': 'Project', 'value': 'MyApp'}]
    #     )
    #     return res['inferenceProfileArn']

    def retrieve_from_db(self, model_id, prompt):

        embedding_model = BedrockEmbeddings(
            client=self.client,
            model_id="amazon.titan-embed-text-v2:0"
        )

        vector_store = Chroma(
            persist_directory="./chroma_store/",
            collection_name="pdf_docs",
            embedding_function=embedding_model
        )

        # Create a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Initialize Bedrock chat model
        chat_model = ChatBedrock(
            model_id=model_id,
            client=self.client,
            provider="mistral"
        )

        collection = vector_store._collection
        document_count = collection.count()
        docs = retriever.get_relevant_documents(prompt)
        breakpoint()

        # Set up RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever
        )

        # Run a sample query
        answer = qa.invoke(prompt)

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

        # 🔽 minimal change: throttle-safe retry wrapper around converse
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
                if code in (
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "RequestLimitExceeded",
                    "ServiceQuotaExceededException",
                ):
                    # exponential backoff + jitter
                    sleep = min(8.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                    time.sleep(sleep)
                    continue
                # non-throttling errors bubble up unchanged
                raise
        # 🔼
