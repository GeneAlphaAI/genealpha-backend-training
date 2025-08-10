from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class TrainingRequest(BaseModel):
    """Request model for starting training"""
    model_type: str = Field(..., description="Type of model to train")
    dataset: str = Field(..., description="Dataset name or path")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Training configuration")
    upload_to_hub: Optional[bool] = Field(default=False, description="Upload to HuggingFace Hub")
    hf_username: Optional[str] = Field(default=None, description="HuggingFace username")

class TrainingResponse(BaseModel):
    """Response model for training request"""
    job_id: str
    status: str
    message: str
    wandb_run_url: Optional[str] = None
