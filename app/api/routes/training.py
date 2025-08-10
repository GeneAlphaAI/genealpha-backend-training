from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from app.schemas.training import TrainingRequest, TrainingResponse
from app.services.training_service import TrainingService
from storage.job_store import JobStore
from storage.model_store import ModelStore
from ml.registry.model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

# Initialize services (in production, use dependency injection)
job_store = JobStore()
model_store = ModelStore()
training_service = TrainingService(job_store, model_store)

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """Start a new training job"""
    
    # Validate model type
    if not ModelRegistry.is_registered(request.model_type):
        available_models = ModelRegistry.list_models()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available models: {', '.join(available_models)}"
        )
    
    # Create job
    job = job_store.create_job(
        model_type=request.model_type,
        dataset=request.dataset,
        config=request.config or {}
    )
    
    # Add upload configuration if specified
    if request.upload_to_hub:
        job.config['upload_to_hub'] = True
        if request.hf_username:
            job.config['hf_username'] = request.hf_username
    
    # Start training in background
    background_tasks.add_task(training_service.train_model_async, job)
    
    return TrainingResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=f"Training job {job.job_id} started successfully",
        wandb_run_url=job.wandb_run_url
    )

@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List all available model types"""
    models = ModelRegistry.list_models()
    return {
        "models": models,
        "total": len(models)
    }
