from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import wandb
import joblib
import os
from datetime import datetime

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.wandb_run = None
        self.trained_at = None
        self.metrics = {}
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the model and return metrics"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def save(self, path: str) -> str:
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'config': self.config,
            'trained_at': self.trained_at,
            'metrics': self.metrics,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
        return path
    
    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.trained_at = model_data.get('trained_at')
        self.metrics = model_data.get('metrics', {})
        return self
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier"""
        pass
    
    def log_to_wandb(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B if run is active"""
        if self.wandb_run and self.wandb_run.id:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
