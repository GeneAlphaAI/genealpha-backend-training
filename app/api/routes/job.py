from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from app.schemas.jobs import JobStatus, JobListResponse, JobStatsResponse
from storage.job_store import JobStore, JobStatus as JobStatusEnum

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Use the same job_store instance
from app.api.routes.training import job_store

@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a specific job"""
    job = job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatus(**job.to_dict())

@router.get("/", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
) -> JobListResponse:
    """List all jobs with optional filtering"""
    
    # Convert status string to enum if provided
    status_enum = None
    if status:
        try:
            status_enum = JobStatusEnum(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {', '.join([s.value for s in JobStatusEnum])}"
            )
    
    # Get jobs
    all_jobs = job_store.list_jobs(status=status_enum)
    
    # Apply pagination
    paginated_jobs = all_jobs[offset:offset + limit]
    
    return JobListResponse(
        total=len(all_jobs),
        jobs=[JobStatus(**job.to_dict()) for job in paginated_jobs]
    )

@router.get("/stats/summary", response_model=JobStatsResponse)
async def get_job_statistics() -> JobStatsResponse:
    """Get summary statistics of all jobs"""
    stats = job_store.get_stats()
    return JobStatsResponse(**stats)

@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel a running job"""
    job = job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status not in [JobStatusEnum.PENDING, JobStatusEnum.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {job.status.value} status"
        )
    
    job.status = JobStatusEnum.CANCELLED
    job_store.update_job(job)
    
    return {"message": f"Job {job_id} cancelled successfully"}
