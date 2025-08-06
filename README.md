# genealpha-backend-training
Backend Template for GeneAlpha Training Engine

```
project_root/
├── README.md
├── pyproject.toml           # or requirements.txt
├── .env                     # HF_API_TOKEN, WANDB_API_KEY, etc.
│
├── config/
│   └── default.yaml         # global defaults (dataset, training, HF, W&B)
│
├── src/
│   ├── main.py              # FastAPI app entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py     # /train, /status, (later) /models, /jobs
│   │
│   ├── config.py            # load YAML/.env into Pydantic settings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_factory.py
│   │         # maps dataset names → HuggingFace Datasets.load_dataset calls
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # abstract BaseModel: fit, predict, save
│   │   ├── linear_regression.py
│   │   ├── random_forest.py
│   │   └── lightgbm.py
│   │         # each implements BaseModel interface and registers itself
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Trainer class: orchestrates data → model → fit → save
│   │   └── job_manager.py    # launches background jobs, tracks status in memory for now
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         # structured logging (e.g. Python logging + file/console)
│       └── wandb_utils.py    # init/project setup, log metrics, artifacts
│
├── scripts/
│   ├── run_server.sh         # uvicorn src.main:app --reload
│   └── run_example.sh        # sample curl/train commands
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_trainer.py
│
└── logs/                     # where logger writes .log files
```
