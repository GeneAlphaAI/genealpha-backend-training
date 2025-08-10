from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import uuid
import json
from threading import Lock

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Job:
    """Represents a training job"""
    
    def __init__(self, model_type: str, dataset: str, config: Dict = None):
        self.job_id = str(uuid.uuid4())
        self.model_type = model_type
        self.dataset = dataset
        self.config = config or {}
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.metrics = {}
        self.wandb_run_id = None
        self.wandb_run_url = None
        self.model_path = None
        self.huggingface_model_id = None
        self.progress = 0
        self.logs = []
    
    def start(self):
        """Mark job as started"""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.add_log(f"Job started at {self.started_at}")
    
    def complete(self, metrics: Dict):
        """Mark job as completed"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.metrics = metrics
        self.progress = 100
        self.add_log(f"Job completed at {self.completed_at}")
    
    def fail(self, error: str):
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        self.add_log(f"Job failed: {error}")
    
    def add_log(self, message: str):
        """Add a log message"""
        self.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'message': message
        })
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'model_type': self.model_type,
            'dataset': self.dataset,
            'config': self.config,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'metrics': self.metrics,
            'wandb_run_id': self.wandb_run_id,
            'wandb_run_url': self.wandb_run_url,
            'model_path': self.model_path,
            'huggingface_model_id': self.huggingface_model_id,
            'progress': self.progress,
            'logs': self.logs
        }


class JobStore:
    """In-memory job storage (will be replaced with PostgreSQL later)"""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()
    
    def create_job(self, model_type: str, dataset: str, config: Dict = None) -> Job:
        """Create a new job"""
        job = Job(model_type, dataset, config)
        with self._lock:
            self._jobs[job.job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job(self, job: Job):
        """Update an existing job"""
        with self._lock:
            self._jobs[job.job_id] = job
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """List all jobs, optionally filtered by status"""
        with self._lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False
    
    def get_stats(self) -> Dict:
        """Get job statistics"""
        with self._lock:
            jobs = list(self._jobs.values())
            return {
                'total': len(jobs),
                'pending': len([j for j in jobs if j.status == JobStatus.PENDING]),
                'running': len([j for j in jobs if j.status == JobStatus.RUNNING]),
                'completed': len([j for j in jobs if j.status == JobStatus.COMPLETED]),
                'failed': len([j for j in jobs if j.status == JobStatus.FAILED])
            }
