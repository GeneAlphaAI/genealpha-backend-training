#!/usr/bin/env python
"""Standalone script to test training without API"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models to trigger registration
import ml.models  # This will register all models
from ml.registry.model_registry import ModelRegistry
from ml.data.loader import DataLoader
from utils.wandb_client import WandbClient
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model type (linear_regression, lightgbm, random_forest)")
    parser.add_argument("--dataset", type=str, default="sample",
                       help="Dataset name or 'sample' for test data")
    parser.add_argument("--config", type=str, default="{}",
                       help="JSON config string")
    
    args = parser.parse_args()
    
    # Import models to register them
    from ml.models import linear_regression, lightgbm_model, random_forest
    
    # Parse config
    config = json.loads(args.config)
    
    print(f"Training {args.model} on {args.dataset}")
    print(f"Config: {config}")
    
    # Initialize components
    data_loader = DataLoader()
    wandb_client = WandbClient()
    
    # Load data
    if args.dataset == "sample":
        X_train, X_val, y_train, y_val = data_loader.create_sample_dataset()
    else:
        # Add your custom data loading logic here
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    # Initialize model
    model = ModelRegistry.get_model(args.model, config)
    
    # Initialize W&B
    run = wandb_client.init_run(
        job_id="manual_run",
        model_type=args.model,
        config=config
    )
    model.wandb_run = run
    
    # Train
    metrics = model.train(X_train, y_train, X_val, y_val)
    print(f"Training completed! Metrics: {metrics}")
    
    # Save model
    model_path = f"./models/{args.model}_manual.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Finish W&B
    wandb_client.finish_run()

if __name__ == "__main__":
    main()
