from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class JobStatus(BaseModel):
    """Job status information"""
    job_id: str
    model_type: str
    dataset: str
    status: str
    progress: int
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    metrics: Dict[str, float]
    wandb_run_url: Optional[str]
    huggingface_model_id: Optional[str]
    logs: List[Dict[str, str]]

class JobListResponse(BaseModel):
    """Response for job listing"""
    total: int
    jobs: List[JobStatus]

class JobStatsResponse(BaseModel):
    """Response for job statistics"""
    total: int
    pending: int
    running: int
    completed: int
    failed: int
