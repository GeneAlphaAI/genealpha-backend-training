import wandb
from typing import Dict, Any, Optional
import os
from datetime import datetime

class WandbClient:
    """Handles W&B experiment tracking"""
    
    def __init__(self, project: str = "ml-training-pipeline", 
                 entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self.run = None
    
    def init_run(self, 
                job_id: str,
                model_type: str,
                config: Dict[str, Any],
                tags: Optional[list] = None,
                notes: Optional[str] = None) -> wandb.Run:
        """Initialize a new W&B run"""
        
        # Initialize W&B run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=f"{model_type}_{job_id[:8]}",
            tags=tags or [model_type, "training"],
            notes=notes or f"Training {model_type} model",
            reinit=True
        )
        
        # Log job metadata
        wandb.config.update({
            "job_id": job_id,
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return self.run
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B"""
        if self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_model(self, model_path: str, model_type: str):
        """Log model artifact to W&B"""
        if self.run:
            artifact = wandb.Artifact(
                name=f"{model_type}_model",
                type="model",
                description=f"Trained {model_type} model"
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def log_dataset(self, dataset_name: str, dataset_info: Dict):
        """Log dataset information to W&B"""
        if self.run:
            wandb.config.update({
                "dataset_name": dataset_name,
                "dataset_info": dataset_info
            })
    
    def finish_run(self, status: str = "success"):
        """Finish the W&B run"""
        if self.run:
            wandb.summary["status"] = status
            self.run.finish()
            self.run = None
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL of the current W&B run"""
        if self.run:
            return self.run.get_url()
        return None
    
    def get_run_id(self) -> Optional[str]:
        """Get the ID of the current W&B run"""
        if self.run:
            return self.run.id
        return None
