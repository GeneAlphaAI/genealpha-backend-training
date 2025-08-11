import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional
from ml.models.base import BaseModel
from ml.registry.model_registry import ModelRegistry
from datetime import datetime

@ModelRegistry.register("random_forest")
class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            max_features=config.get('max_features', 'sqrt'),
            bootstrap=config.get('bootstrap', True),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
    
    @property
    def model_type(self) -> str:
        return "random_forest"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Random Forest model"""
        self.trained_at = datetime.utcnow()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        # Calculate validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = {
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            self.metrics = {**train_metrics, **val_metrics}
        else:
            self.metrics = train_metrics
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {f'feature_{i}_importance': imp 
                             for i, imp in enumerate(self.model.feature_importances_)}
            self.metrics.update(importance_dict)
        
        # Log to W&B if active
        self.log_to_wandb(self.metrics)
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
