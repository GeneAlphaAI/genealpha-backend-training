import os
import shutil
from typing import Optional
from pathlib import Path

class ModelStore:
    """Handles model artifact storage"""
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, job_id: str, model_type: str) -> Path:
        """Get the path for storing a model"""
        model_dir = self.base_path / model_type / job_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / "model.pkl"
    
    def save_model(self, model, job_id: str, model_type: str) -> str:
        """Save a model and return the path"""
        model_path = self.get_model_path(job_id, model_type)
        model.save(str(model_path))
        return str(model_path)
    
    def load_model(self, job_id: str, model_type: str):
        """Load a saved model"""
        from ml.registry.model_registry import ModelRegistry
        
        model_path = self.get_model_path(job_id, model_type)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = ModelRegistry.get_model(model_type)
        model.load(str(model_path))
        return model
    
    def delete_model(self, job_id: str, model_type: str) -> bool:
        """Delete a saved model"""
        model_dir = self.base_path / model_type / job_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False
    
    def list_models(self) -> list:
        """List all saved models"""
        models = []
        for model_type_dir in self.base_path.iterdir():
            if model_type_dir.is_dir():
                for job_dir in model_type_dir.iterdir():
                    if job_dir.is_dir():
                        model_path = job_dir / "model.pkl"
                        if model_path.exists():
                            models.append({
                                'model_type': model_type_dir.name,
                                'job_id': job_dir.name,
                                'path': str(model_path),
                                'size': model_path.stat().st_size
                            })
        return models
