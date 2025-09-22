from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from base import ModelCategory


class HyperparameterSchema(BaseModel):
    """Schema for hyperparameter definition"""
    name: str
    type: str  # "int", "float", "string", "bool", "list"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: Optional[str] = None

class ModelInfo(BaseModel):
    """Information about available models"""
    model_type: str
    name: str
    description: str
    category: ModelCategory
    supports: Dict[str, bool] = {
        "multiclass": True,
        "multilabel": False,
        "predict_proba": True,
        "feature_importance": True
    }
    default_hyperparameters: Dict[str, Any]
    hyperparameter_schema: List[HyperparameterSchema]
    required_packages: List[str]
    estimated_training_time: Optional[str] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total: int