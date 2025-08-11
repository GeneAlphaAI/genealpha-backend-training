import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional
from ml.models.base import BaseModel
from ml.registry.model_registry import ModelRegistry
from datetime import datetime

@ModelRegistry.register("lightgbm")
class LightGBMModel(BaseModel):
    """LightGBM implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.lgb_params = {
            'objective': config.get('objective', 'regression'),
            'metric': config.get('metric', 'rmse'),
            'boosting_type': config.get('boosting_type', 'gbdt'),
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': config.get('verbose', -1),
            'random_state': config.get('random_state', 42)
        }
        self.num_boost_round = config.get('num_boost_round', 100)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
    
    @property
    def model_type(self) -> str:
        return "lightgbm"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train LightGBM model"""
        self.trained_at = datetime.utcnow()
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        # Training with callbacks for W&B logging
        callbacks = []
        if self.wandb_run:
            def wandb_callback(env):
                for name, metric, value in env.evaluation_result_list:
                    self.log_to_wandb({f"{name}_{metric}": value}, step=env.iteration)
            callbacks.append(wandb_callback)
        
        # Train model
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks if callbacks else None,
            init_model=None,
            keep_training_booster=False
        )
        
        # Calculate final metrics
        train_pred = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        train_metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
            val_metrics = {
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            self.metrics = {**train_metrics, **val_metrics}
        else:
            self.metrics = train_metrics
        
        self.metrics['best_iteration'] = self.model.best_iteration
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
