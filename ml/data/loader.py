from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataLoader:
    """Handles loading datasets from HuggingFace and local sources"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.target_column = None
    
    def load_from_huggingface(self, dataset_name: str, 
                             split: str = "train",
                             config_name: Optional[str] = None) -> Dataset:
        """Load dataset from HuggingFace Hub"""
        try:
            dataset = load_dataset(dataset_name, config_name, split=split)
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_name}: {str(e)}")
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        return pd.read_csv(file_path)
    
    def prepare_tabular_data(self, dataset, 
                           target_column: str,
                           feature_columns: Optional[list] = None,
                           test_size: float = 0.2,
                           scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare tabular data for training"""
        
        # Convert to pandas DataFrame if it's a HuggingFace dataset
        if hasattr(dataset, 'to_pandas'):
            df = dataset.to_pandas()
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise ValueError("Dataset must be a HuggingFace Dataset or pandas DataFrame")
        
        # Store column information
        self.target_column = target_column
        
        # Determine feature columns
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        # Extract features and target
        X = df[self.feature_columns].values
        y = df[target_column].values
        
        # Handle categorical features
        X = self._handle_categorical_features(X, df[self.feature_columns])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        return X_train, X_val, y_train, y_val
    
    def _handle_categorical_features(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Handle categorical features in the dataset"""
        processed_cols = []
        
        for i, col in enumerate(df.columns):
            if df[col].dtype == 'object':
                # Use label encoding for categorical columns
                le = LabelEncoder()
                processed_cols.append(le.fit_transform(df[col]))
            else:
                processed_cols.append(df[col].values)
        
        return np.column_stack(processed_cols) if processed_cols else X
    
    def create_sample_dataset(self, n_samples: int = 1000, 
                            n_features: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create a sample dataset for testing"""
        from sklearn.datasets import make_regression
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=10,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val
