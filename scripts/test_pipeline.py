#!/usr/bin/env python
"""Test the complete pipeline"""

import requests
import time
import json

API_URL = "http://localhost:8000/api/v1"

def test_pipeline():
    print("Testing ML Training Pipeline...")
    
    # 1. Check health
    print("\n1. Checking health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Health: {response.json()}")
    
    # 2. List available models
    print("\n2. Listing available models...")
    response = requests.get(f"{API_URL}/training/models")
    models = response.json()
    print(f"Available models: {models}")
    
    # 3. Start a training job
    print("\n3. Starting training job...")
    training_request = {
        "model_type": "random_forest",
        "dataset": "sample",
        "config": {
            "n_estimators": 50,
            "max_depth": 10,
            "n_samples": 500,
            "n_features": 8
        }
    }
    
    response = requests.post(
        f"{API_URL}/training/start",
        json=training_request
    )
    job_info = response.json()
    job_id = job_info["job_id"]
    print(f"Job started: {job_info}")
    
    # 4. Monitor job status
    print("\n4. Monitoring job status...")
    for i in range(30):  # Check for 30 seconds
        response = requests.get(f"{API_URL}/jobs/{job_id}")
        job_status = response.json()
        print(f"Status: {job_status['status']}, Progress: {job_status['progress']}%")
        
        if job_status["status"] in ["completed", "failed"]:
            break
        
        time.sleep(2)
    
    # 5. Get final results
    print("\n5. Final job details:")
    response = requests.get(f"{API_URL}/jobs/{job_id}")
    final_status = response.json()
    print(json.dumps(final_status, indent=2))
    
    # 6. List all jobs
    print("\n6. Listing all jobs...")
    response = requests.get(f"{API_URL}/jobs/")
    jobs = response.json()
    print(f"Total jobs: {jobs['total']}")
    
    # 7. Get job statistics
    print("\n7. Job statistics...")
    response = requests.get(f"{API_URL}/jobs/stats/summary")
    stats = response.json()
    print(f"Stats: {stats}")
    
    print("\nPipeline test completed!")

if __name__ == "__main__":
    test_pipeline()
