# GeneAlpha ML Training Pipeline

A modular, scalable machine learning training pipeline with FastAPI, W&B experiment tracking, and HuggingFace Hub integration.

## Project Structure

```
ml-training-pipeline/
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   └── core/              # Core configuration
├── ml/                    # ML components
│   ├── models/            # Model implementations
│   ├── data/              # Data loading/preprocessing
│   └── registry/          # Model registry system
├── storage/               # Storage components
├── utils/                 # Utility functions
├── configs/               # Configuration files
└── scripts/               # Helper scripts
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-training-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - WANDB_API_KEY (get from https://wandb.ai/settings)
# - HUGGINGFACE_TOKEN (get from https://huggingface.co/settings/tokens)
```

### 3. Start the Server

```bash
# Using the startup script
chmod +x start_server.sh
./start_server.sh

# Or manually
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Training Endpoints

#### Start Training
```bash
POST /api/v1/training/start

# Example request
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "dataset": "sample",
    "config": {
      "n_estimators": 100,
      "max_depth": 10
    }
  }'
```

#### List Available Models
```bash
GET /api/v1/training/models
```

### Job Management

#### Get Job Status
```bash
GET /api/v1/jobs/{job_id}
```

#### List All Jobs
```bash
GET /api/v1/jobs/
```

#### Get Job Statistics
```bash
GET /api/v1/jobs/stats/summary
```

## Usage Examples

### 1. Train a Model Using API

```python
import requests

# Start training
response = requests.post(
    "http://localhost:8000/api/v1/training/start",
    json={
        "model_type": "lightgbm",
        "dataset": "sample",
        "config": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_samples": 1000
        }
    }
)

job = response.json()
print(f"Job ID: {job['job_id']}")
```

### 2. Using Custom Datasets

#### From HuggingFace
```python
{
    "model_type": "linear_regression",
    "dataset": "scikit-learn/diabetes",  # HuggingFace dataset
    "config": {
        "target_column": "target",
        "test_size": 0.2
    }
}
```

#### From Local CSV
```python
{
    "model_type": "random_forest",
    "dataset": "./data/my_data.csv",  # Local file
    "config": {
        "target_column": "price",
        "feature_columns": ["size", "rooms", "location"]
    }
}
```

### 3. Standalone Training Script

```bash
python scripts/run_training.py \
  --model lightgbm \
  --dataset sample \
  --config '{"num_leaves": 50}'
```

## Adding New Models

Adding a new model is simple:

1. Create a new file in `ml/models/`:

```python
# ml/models/xgboost_model.py
from ml.models.base import BaseModel
from ml.registry.model_registry import ModelRegistry
import xgboost as xgb

@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    @property
    def model_type(self) -> str:
        return "xgboost"
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Training implementation
        pass
    
    def predict(self, X):
        # Prediction implementation
        pass
```

2. Import it in `ml/models/__init__.py`:
```python
from ml.models.xgboost_model import XGBoostModel
```

3. The model is now automatically available via the API!

## Testing

### Run the Test Script
```bash
python scripts/test_pipeline.py
```

This will:
1. Check health status
2. List available models
3. Start a training job
4. Monitor progress
5. Display results

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Start training
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_type": "linear_regression", "dataset": "sample"}'

# Check job status (replace JOB_ID)
curl http://localhost:8000/api/v1/jobs/JOB_ID
```

## W&B Integration

Training runs are automatically tracked in W&B. View your experiments at:
https://wandb.ai/YOUR_USERNAME/ml-training-pipeline

Each run logs:
- Training/validation metrics
- Model configuration
- Training progress
- Model artifacts

## HuggingFace Hub Integration

To upload models to HuggingFace Hub:

```python
{
    "model_type": "random_forest",
    "dataset": "sample",
    "upload_to_hub": true,
    "hf_username": "your-username"
}
```

Models will be uploaded to:
`https://huggingface.co/your-username/MODEL_TYPE-JOB_ID`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **W&B Authentication**: Set your API key:
   ```bash
   export WANDB_API_KEY=your_key_here
   ```

3. **Port Already in Use**: Change the port:
   ```bash
   uvicorn app.main:app --port 8001
   ```

4. **Model Not Found**: Check that the model is registered:
   ```python
   from ml.registry.model_registry import ModelRegistry
   print(ModelRegistry.list_models())
   ```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Database Integration

To integrate PostgreSQL (future enhancement):

1. Install dependencies:
   ```bash
   pip install sqlalchemy psycopg2-binary
   ```

2. Update `storage/job_store.py` to use SQLAlchemy models

3. Configure database URL in `.env`:
   ```
   DATABASE_URL=postgresql://user:password@localhost/mlpipeline
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model/feature
4. Write tests
5. Submit a pull request


## Support

For issues or questions:
1. Check the documentation at `/docs`
2. Review existing issues on GitHub
3. Create a new issue with details

## Roadmap

- [ ] PostgreSQL integration
- [ ] Authentication/Authorization
- [ ] Model serving endpoints
- [ ] Integretion of Deep Learning Models
- [ ] AutoML capabilities
- [ ] Model monitoring/drift detection
