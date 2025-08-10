from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import os

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ML Training Pipeline"
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for the service"""
    # Check if required services are available
    checks = {
        "wandb": _check_wandb(),
        "huggingface": _check_huggingface(),
        "storage": _check_storage()
    }
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

def _check_wandb() -> bool:
    """Check if W&B is configured"""
    try:
        import wandb
        return wandb.api.api_key is not None or os.getenv("WANDB_API_KEY") is not None
    except:
        return False

def _check_huggingface() -> bool:
    """Check if HuggingFace is configured"""
    return os.getenv("HUGGINGFACE_TOKEN") is not None

def _check_storage() -> bool:
    """Check if storage directories are accessible"""
    try:
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        return True
    except:
        return False
