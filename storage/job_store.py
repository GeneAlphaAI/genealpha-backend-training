from __future__ import annotations

import uuid
import logging
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Job:
    """
    Represents a single training run.
    """

    def __init__(
        self,
        model_type: str,
        dataset: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        user_id: Optional[str] = None,
        precreated_job_id: Optional[str] = None,
    ) -> None:
        self.job_id: str = precreated_job_id or str(uuid.uuid4())
        self.user_id: Optional[str] = user_id
        self.model_type: str = model_type
        self.dataset: str = dataset
        self.config: Dict[str, Any] = config or {}

        # runtime state 
        self.status: JobStatus = JobStatus.PENDING
        self.created_at: datetime = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.wandb_run_id: Optional[str] = None
        self.wandb_run_url: Optional[str] = None
        self.model_path: Optional[str] = None
        self.huggingface_model_id: Optional[str] = None  # populated on upload
        self.progress: int = 0
        self.logs: List[Dict[str, str]] = []

    # state mutators
    def start(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.add_log(f"Job started at {self.started_at.isoformat()}")

    def complete(
        self,
        *,
        metrics: Dict[str, Any],
        huggingface_model_id: Optional[str],
    ) -> None:
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.metrics = metrics
        self.huggingface_model_id = huggingface_model_id
        self.progress = 100
        self.add_log(f"Job completed at {self.completed_at.isoformat()}")

    def fail(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        self.add_log(f"Job failed: {error}")

    def add_log(self, message: str) -> None:
        self.logs.append(
            {"timestamp": datetime.utcnow().isoformat(), "message": str(message)}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current job state."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "model_type": self.model_type,
            "dataset": self.dataset,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error": self.error,
            "metrics": self.metrics,
            "wandb_run_id": self.wandb_run_id,
            "wandb_run_url": self.wandb_run_url,
            "model_path": self.model_path,
            "huggingface_model_id": self.huggingface_model_id,
            "progress": self.progress,
            "logs": self.logs,
        }


class JobStore:
    """
    **Thread-safe singleton** that keeps all Job objects in memory.
    Swap this implementation for Postgres / Redis later .
    """

    _instance: Optional["JobStore"] = None

    @classmethod
    def instance(cls) -> "JobStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock: Lock = Lock()

    def create_job(
        self,
        model_type: str,
        dataset: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        **Legacy helper** – creates & registers a new job and returns it.
        """
        job = Job(model_type, dataset, config)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def add(
        self,
        job_id: str,
        model_type: str,
        dataset: str,
        user_id: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Used by the asynchronous `TrainingService.start_job()`:
        """
        job = Job(
            model_type,
            dataset,
            extra.get("config") if extra else {},
            user_id=user_id,
            precreated_job_id=job_id,
        )
        with self._lock:
            self._jobs[job_id] = job

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in fields.items():
                setattr(job, k, v)

    def complete(
        self,
        job_id: str,
        *,
        metrics: Dict[str, Any],
        huggingface_model_id: Optional[str],
    ) -> None:
        with self._lock:
            self._jobs[job_id].complete(
                metrics=metrics, huggingface_model_id=huggingface_model_id
            )

    def fail(self, job_id: str, *, error: str) -> None:
        with self._lock:
            self._jobs[job_id].fail(error)

    # read-only helpers 
    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(
        self,
        *,
        status: Optional[JobStatus] = None,
        user_id: Optional[str] = None,
    ) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
            if status is not None:
                jobs = [j for j in jobs if j.status == status]
            if user_id is not None:
                jobs = [j for j in jobs if j.user_id == user_id]
            return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._jobs)
            return {
                "total": total,
                "pending": len([j for j in self._jobs.values() if j.status == JobStatus.PENDING]),
                "running": len([j for j in self._jobs.values() if j.status == JobStatus.RUNNING]),
                "completed": len(
                    [j for j in self._jobs.values() if j.status == JobStatus.COMPLETED]
                ),
                "failed": len([j for j in self._jobs.values() if j.status == JobStatus.FAILED]),
            }
