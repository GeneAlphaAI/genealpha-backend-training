import asyncio
from typing import Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from ml.registry.model_registry import ModelRegistry
from ml.data.loader import DataLoader
from storage.job_store import Job, JobStatus
from storage.model_store import ModelStore
from utils.wandb_client import WandbClient
from utils.huggingface_client import HuggingFaceClient

logger = logging.getLogger(__name__)

class TrainingService:
    """Handles model training orchestration"""
    
    def __init__(self, job_store, model_store: ModelStore):
        self.job_store = job_store
        self.model_store = model_store
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.data_loader = DataLoader()
        self.wandb_client = WandbClient()
        self.hf_client = HuggingFaceClient()
    
    async def train_model_async(self, job: Job):
        """Train a model asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run training in thread pool to avoid blocking
        try:
            await loop.run_in_executor(
                self.executor,
                self._train_model_sync,
                job
            )
        except Exception as e:
            logger.error(f"Training failed for job {job.job_id}: {str(e)}")
            job.fail(str(e))
            self.job_store.update_job(job)
    
    def _train_model_sync(self, job: Job):
        """Synchronous model training logic"""
        try:
            # Start the job
            job.start()
            self.job_store.update_job(job)
            logger.info(f"Starting training for job {job.job_id}")
            
            # Initialize W&B run
            job.add_log("Initializing W&B tracking...")
            wandb_run = self.wandb_client.init_run(
                job_id=job.job_id,
                model_type=job.model_type,
                config=job.config
            )
            job.wandb_run_id = self.wandb_client.get_run_id()
            job.wandb_run_url = self.wandb_client.get_run_url()
            job.progress = 10
            self.job_store.update_job(job)
            
            # Load and prepare data
            job.add_log(f"Loading dataset: {job.dataset}")
            X_train, X_val, y_train, y_val = self._load_data(job.dataset, job.config)
            job.progress = 30
            self.job_store.update_job(job)
            
            # Initialize model
            job.add_log(f"Initializing {job.model_type} model...")
            model = ModelRegistry.get_model(job.model_type, job.config)
            model.wandb_run = wandb_run
            job.progress = 40
            self.job_store.update_job(job)
            
            # Train model
            job.add_log("Training model...")
            metrics = model.train(X_train, y_train, X_val, y_val)
            job.progress = 80
            self.job_store.update_job(job)
            
            # Save model locally
            job.add_log("Saving model artifacts...")
            model_path = self.model_store.save_model(model, job.job_id, job.model_type)
            job.model_path = model_path
            
            # Log model to W&B
            self.wandb_client.log_model(model_path, job.model_type)
            
            # Upload to HuggingFace Hub if configured
            if job.config.get('upload_to_hub', False):
                job.add_log("Uploading model to HuggingFace Hub...")
                repo_id = f"{job.config.get('hf_username', 'user')}/{job.model_type}-{job.job_id[:8]}"
                model_card = {
                    'model_type': job.model_type,
                    'job_id': job.job_id,
                    'trained_at': model.trained_at.isoformat() if model.trained_at else None,
                    'metrics': metrics,
                    'config': job.config
                }
                
                self.hf_client.create_model_repo(repo_id)
                model_url = self.hf_client.upload_model(model_path, repo_id, model_card=model_card)
                job.huggingface_model_id = repo_id
            
            # Finish W&B run
            self.wandb_client.finish_run()
            
            # Complete the job
            job.complete(metrics)
            job.add_log("Training completed successfully!")
            self.job_store.update_job(job)
            
            logger.info(f"Training completed for job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Training error for job {job.job_id}: {str(e)}")
            job.fail(str(e))
            self.job_store.update_job(job)
            
            # Clean up W&B run
            if self.wandb_client.run:
                self.wandb_client.finish_run(status="failed")
    
    def _load_data(self, dataset_name: str, config: Dict) -> tuple:
        """Load and prepare dataset"""
        # Check if it's a sample dataset
        if dataset_name == "sample":
            return self.data_loader.create_sample_dataset(
                n_samples=config.get('n_samples', 1000),
                n_features=config.get('n_features', 10)
            )
        
        # Check if it's a local CSV file
        if dataset_name.endswith('.csv'):
            df = self.data_loader.load_from_csv(dataset_name)
            return self.data_loader.prepare_tabular_data(
                df,
                target_column=config.get('target_column', 'target'),
                feature_columns=config.get('feature_columns'),
                test_size=config.get('test_size', 0.2)
            )
        
        # Load from HuggingFace
        dataset = self.data_loader.load_from_huggingface(
            dataset_name,
            split=config.get('split', 'train'),
            config_name=config.get('config_name')
        )
        
        return self.data_loader.prepare_tabular_data(
            dataset,
            target_column=config.get('target_column', 'label'),
            feature_columns=config.get('feature_columns'),
            test_size=config.get('test_size', 0.2)
        )
