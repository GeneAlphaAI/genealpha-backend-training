# Auto-import models to register them
from ml.models.linear_regression import LinearRegressionModel
from ml.models.lightgbm_model import LightGBMModel
from ml.models.random_forest import RandomForestModel

__all__ = ['LinearRegressionModel', 'LightGBMModel', 'RandomForestModel']
