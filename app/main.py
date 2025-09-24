from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import training, job, health, datasets
from utils.logging import setup_logging
import logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router, prefix=f"{settings.api_prefix}")
app.include_router(job.router, prefix=f"{settings.api_prefix}")
app.include_router(health.router, prefix=f"{settings.api_prefix}")
app.include_router(datasets.router, prefix=f"{settings.api_prefix}") 

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Training Pipeline API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health"
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Import models to register them
    from ml.models import linear_regression, lightgbm_model, random_forest
    from ml.registry.model_registry import ModelRegistry
    
    available_models = ModelRegistry.list_models()
    logger.info(f"Registered models: {', '.join(available_models)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down ML Training Pipeline")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers
    )
