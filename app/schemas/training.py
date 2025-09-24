from pydantic import BaseModel, Field, constr   
from typing import Optional, Dict, Any
from datetime import datetime

class TrainingRequest(BaseModel):
    """Request model for starting training"""
    user_id: constr(pattern=r"^[a-zA-Z0-9_\-]+$")
    model_type: str = Field(..., description="Type of model to train")
    dataset: str = Field(..., description="Dataset name or path")
    config: Dict[str, Any] = {}
    upload_to_hub: Optional[bool] = Field(default=False, description="Upload to HuggingFace Hub")


class TrainingResponse(BaseModel):
    """Response model for training request"""
    job_id: str
    status: str
    message: str
    created_at: datetime
    wandb_run_url: Optional[str] = None
