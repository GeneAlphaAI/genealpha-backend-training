import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "ML Training Pipeline"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # W&B Configuration
    wandb_project: str = "ml-training-pipeline"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # HuggingFace Configuration
    huggingface_token: Optional[str] = None
    
    # Storage Configuration
    models_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    
    # Training Configuration
    max_concurrent_jobs: int = 4
    default_batch_size: int = 32
    default_epochs: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
