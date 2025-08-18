# Import all model classes to trigger automatic registration via decorators
from ml.models.base import BaseModel
from ml.models.linear_regression import LinearRegressionModel
from ml.models.lightgbm_model import LightGBMModel
from ml.models.random_forest import RandomForestModel

__all__ = ['BaseModel', 'LinearRegressionModel', 'LightGBMModel', 'RandomForestModel']
