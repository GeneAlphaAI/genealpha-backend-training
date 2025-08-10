# genealpha-backend-training
Backend Template for GeneAlpha Training Engine

```
genealpha-backend-training/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── training.py     # Training endpoints
│   │   │   ├── jobs.py         # Job status endpoints
│   │   │   └── health.py       # Health check endpoints
│   │   └── dependencies.py     # Shared dependencies
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── training.py         # Pydantic models for requests/responses
│   │   └── jobs.py             # Job-related schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── job_manager.py      # Job orchestration logic
│   │   └── training_service.py # Training coordination
│   │
│   └── core/
│       ├── __init__.py
│       ├── config.py           # Application configuration
│       └── exceptions.py       # Custom exceptions
│
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base model class
│   │   ├── linear_regression.py
│   │   ├── lightgbm_model.py
│   │   └── random_forest.py
│   │
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py     # Abstract trainer class
│   │   └── model_trainer.py    # Concrete trainer implementation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Dataset loading from HuggingFace
│   │   └── preprocessor.py     # Data preprocessing utilities
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Model evaluation metrics
│   │
│   └── registry/
│       ├── __init__.py
│       └── model_registry.py   # Model registration system
│
├── storage/
│   ├── __init__.py
│   ├── job_store.py            # In-memory job storage (temp)
│   └── model_store.py          # Model artifacts storage
│
├── utils/
│   ├── __init__.py
│   ├── logging.py              # Logging configuration
│   ├── wandb_client.py         # W&B integration
│   └── huggingface_client.py   # HuggingFace Hub integration
│
├── configs/
│   ├── model_configs.yaml      # Model-specific configurations
│   └── training_configs.yaml   # Training hyperparameters
│
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   ├── test_api/
│   └── test_trainers/
│
├── scripts/
│   ├── run_training.py         # Standalone training script
│   └── test_pipeline.py        # Pipeline testing script
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml          # For future containerization
```

Training Pipeline Flow

```
User Request → FastAPI Endpoint → Job Manager → Training Service
                                                        ↓
                                                 Model Trainer
                                                        ↓
                                              [Data Loading (HF)]
                                                        ↓
                                              [Model Training]
                                                        ↓
                                         [W&B Logging + HF Upload]
                                                        ↓
                                              Job Status Update
```
