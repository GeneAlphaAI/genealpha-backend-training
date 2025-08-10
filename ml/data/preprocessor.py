import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Handles data preprocessing tasks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}
    
    def handle_missing_values(self, data: np.ndarray, 
                            strategy: str = 'mean',
                            column_name: Optional[str] = None) -> np.ndarray:
        """Handle missing values in the data"""
        imputer = SimpleImputer(strategy=strategy)
        data_imputed = imputer.fit_transform(data.reshape(-1, 1) if data.ndim == 1 else data)
        
        if column_name:
            self.imputers[column_name] = imputer
        
        return data_imputed
    
    def scale_features(self, data: np.ndarray, 
                      method: str = 'standard',
                      column_name: Optional[str] = None) -> np.ndarray:
        """Scale features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        data_scaled = scaler.fit_transform(data.reshape(-1, 1) if data.ndim == 1 else data)
        
        if column_name:
            self.scalers[column_name] = scaler
        
        return data_scaled
    
    def remove_outliers(self, data: np.ndarray, 
                       threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using z-score method"""
        from scipy import stats
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < threshold]
    
    def encode_categorical(self, data: pd.Series, 
                         method: str = 'label') -> np.ndarray:
        """Encode categorical variables"""
        if method == 'label':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            return encoder.fit_transform(data)
        elif method == 'onehot':
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            return encoder.fit_transform(data.values.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic feature engineering"""
        # Add polynomial features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        return df
