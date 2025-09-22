from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from base import JobStatus


class JobStatusResponse(BaseModel):
    """Detailed job status"""
    job_id: str
    user_id: str
    status: JobStatus
    model_type: str
    dataset_id: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    metrics: Optional[Dict[str, float]] = None
    model_url: Optional[str] = None
    wandb_run_url: Optional[str] = None
    error_message: Optional[str] = None

class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total: int
    page: int
    per_page: int