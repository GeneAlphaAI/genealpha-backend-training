from typing import Dict, Type, List
from ml.models.base import BaseModel

class ModelRegistry:
    """Registry for auto-discovering and managing ML models"""
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """Decorator to register models"""
        def decorator(model_class: Type[BaseModel]):
            cls._models[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, model_type: str, config: Dict[str, Any] = None) -> BaseModel:
        """Get an instance of a registered model"""
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{model_type}' not registered. Available: {available}")
        
        config = config or {}
        return cls._models[model_type](config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model types"""
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered"""
        return model_type in cls._models
