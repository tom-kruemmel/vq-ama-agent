# src/config.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str = "us-east-1"
    bedrock_model_id: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage:
# settings = Settings()
# print(settings.aws_access_key_id)
