"""
Script to upload a CSV dataset to a private Hugging Face Hub repository
"""
import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login, create_repo

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")

HF_USERNAME  = os.getenv("HF_USERNAME", "your_username")  
DATASET_NAME = "my_csv_dataset"                          # desired repo name
CSV_PATH     = "path/to/your_dataset.csv"               

login(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)

# Single-split dataset; key "train" can be renamed if desired
ds = load_dataset("csv", data_files={"train": CSV_PATH})["train"]

ds.push_to_hub(repo_id=repo_id, private=True)
print(f"✅ Uploaded private dataset to https://huggingface.co/datasets/{repo_id}")
