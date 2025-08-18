from typing import Dict, Type, List, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ml.models.base import BaseModel

class ModelRegistry:
    """Registry for auto-discovering and managing ML models"""
    _models: Dict[str, Type["BaseModel"]] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """Decorator to register a model class with the registry."""
        def decorator(model_class: Type["BaseModel"]) -> Type["BaseModel"]:
            # Lazy import to avoid circular imports
            from ml.models.base import BaseModel
            
            if not issubclass(model_class, BaseModel):
                raise ValueError(f"Model class {model_class.__name__} must inherit from BaseModel")
            
            cls._models[model_type] = model_class
            logging.info(f"Registered model: {model_type}")
            return model_class
        return decorator
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type["BaseModel"]):
        """Register a model class manually"""
        # Lazy import to avoid circular imports
        from ml.models.base import BaseModel
        
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class {model_class.__name__} must inherit from BaseModel")
        
        cls._models[model_type] = model_class
        logging.info(f"Registered model: {model_type}")
    
    @classmethod
    def get_model(cls, model_type: str, config: Dict[str, Any] = None):
        """Get an instance of a registered model"""
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{model_type}' not registered. Available: {available}")
        
        config = config or {}
        return cls._models[model_type](config)
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type["BaseModel"]:
        """Get a model class (not instance) by name"""
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{model_type}' not registered. Available: {available}")
        return cls._models[model_type]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model types"""
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered"""
        return model_type in cls._models
