from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from base import JobStatus

class TrainingConfig(BaseModel):
    """Training configuration with validation"""
    features: List[str] = Field(..., min_items=1, description="Feature columns")
    target_column: str = Field(..., description="Target column")
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    random_seed: int = Field(42, ge=0, le=2**32-1)
    
    # Model hyperparameters
    params: Dict[str, Any] = Field(default_factory=dict)
    
    # Output settings
    experiment_name: Optional[str] = None
    push_to_hub: bool = Field(True)
    repo_visibility: Literal["public", "private"] = Field("private")
    save_locally: bool = Field(True)
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("At least one feature must be selected")
        if len(set(v)) != len(v):
            raise ValueError("Duplicate features found")
        return v
    
    @validator('target_column')
    def target_not_in_features(cls, v, values):
        if 'features' in values and v in values['features']:
            raise ValueError("target_column cannot be in features")
        return v
    

class TrainingRequest(BaseModel):
    """Request model for starting training"""
    user_id: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(..., description="Type of model to train")
    dataset_id: str = Field(..., description="Dataset identifier")
    config: TrainingConfig
    
    @validator('user_id')
    def sanitize_user_id(cls, v):
        # Remove special characters for HF repo naming
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', v)
    

class TrainingResponse(BaseModel):
    """Response after starting a training job"""
    job_id: str
    user_id: str
    status: JobStatus
    message: str
    created_at: datetime
    wandb_run_url: Optional[str] = None