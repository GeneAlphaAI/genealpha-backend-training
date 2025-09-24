from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status

from app.schemas.training import TrainingRequest, TrainingResponse
from app.services.training_service import TrainingService
from storage.job_store import JobStore
from ml.registry.model_registry import ModelRegistry

import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


job_store = JobStore()   
training_service = TrainingService(job_store)

@router.post(
    "/start",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_training(request: TrainingRequest) -> TrainingResponse:

    if not ModelRegistry.is_registered(request.model_type):
        available = ", ".join(ModelRegistry.list_models())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available models: {available}",
        )

    try:
        job_id = training_service.start_job(request)
    except ValueError as e:      
        raise HTTPException(status_code=400, detail=str(e)) from e

    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message=f"Training job {job_id} queued.",
        created_at=datetime.utcnow(),
    )

@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """
    List all registered trainer implementations.
    """
    models = ModelRegistry.list_models()
    return {"models": models, "total": len(models)}
