from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from base import DataFeatureType

class DataFeatureDesc(BaseModel):
    name: str
    dtype: DataFeatureType
    nullable: bool = False
    unique_values: Optional[int] = None
    min: Optional[float] = None
    max: Optional[float] = None

class DatasetInfo(BaseModel):
    """Information about available datasets"""
    dataset_id: str
    name: str
    description: Optional[str] = None
    size: str
    features: List[DataFeatureDesc]
    target_columns: List[DataFeatureDesc]
    sample_count: int
    splits: Optional[List[str]] = None
    license: Optional[str] = None
    revision: Optional[str] = None
    last_updated: datetime
    hf_path: str  # Full HuggingFace path

class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]
    total: int