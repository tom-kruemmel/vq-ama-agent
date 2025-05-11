import os
import json
import boto3
from typing import Dict, Any, List

class BedrockClient:
    """
    Simple wrapper around the AWS Bedrock Runtime API.
    """

    def __init__(self):
        # Use the Bedrock Runtime endpoint for inference
        self.client = boto3.client('bedrock-runtime')  # :contentReference[oaicite:0]{index=0}

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available foundation models in the current region.
        """
        # Some regions use 'bedrock' rather than 'bedrock-runtime' for listing
        meta_client = boto3.client('bedrock')
        response = meta_client.list_foundation_models()
        return response.get('foundationModels', [])

    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Invoke a text model to generate a completion.
        """
        payload = {
            'prompt': prompt,
            'max_tokens_to_sample': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        # Streaming version also available via invoke_model_with_response_stream
        response = self.client.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )  # :contentReference[oaicite:1]{index=1}

        # The API returns a streaming Body, but for small prompts it's buffered
        body_bytes = response['body'].read()
        return json.loads(body_bytes)

# Example usage:
if __name__ == "__main__":
    bc = BedrockClient()
    print("Available models:")
    for m in bc.list_models():
        print(f" - {m['modelId']}: {m.get('modelArn')}")

    # Simple completion
    resp = bc.invoke_model(
        model_id="amazon.titan-text", 
        prompt="Q: What is RAG? A:",
        max_tokens=200
    )
    print("Model response:", resp.get('results'))
