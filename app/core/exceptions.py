class TrainingPipelineException(Exception):
    """Base exception for training pipeline"""
    pass

class ModelNotFoundError(TrainingPipelineException):
    """Raised when model is not found"""
    pass

class DatasetLoadError(TrainingPipelineException):
    """Raised when dataset cannot be loaded"""
    pass

class TrainingError(TrainingPipelineException):
    """Raised when training fails"""
    pass

class JobNotFoundError(TrainingPipelineException):
    """Raised when job is not found"""
    pass
