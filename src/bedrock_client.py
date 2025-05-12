import os
import json
import boto3
from typing import Dict, Any, List

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
        self.client = session.client('bedrock-runtime')

    
    def get_account_id(self) -> str:
        sts = boto3.Session().client('sts')
        return sts.get_caller_identity()['Account']

    def list_models(self) -> List[Dict[str, Any]]:
        # Some regions require the 'bedrock' meta-client
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
        # payload = {
        #     'prompt': prompt,
        #     'max_tokens_to_sample': max_tokens,
        #     'temperature': temperature,
        #     'top_p': top_p,
        # }

        payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 1.0
            }
        }
        resp = self.client.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload),
        )
        body = resp['body'].read()
        return json.loads(body)
