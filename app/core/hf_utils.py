import logging, time, uuid, os
from datetime import datetime
from huggingface_hub import HfApi, upload_folder, hf_hub_download
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:              
    from huggingface_hub.utils import RepositoryNotFoundError as HfHubHTTPError

log = logging.getLogger(__name__)
HF_ORG = "GeneAlpha"
_api = HfApi()

def build_repo_id(user_id: str, model_key: str, dataset_key: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{HF_ORG}/{user_id}_{model_key}_{dataset_key.replace('/', '-') }__{ts}-{short}"

def ensure_dataset_exists(dataset: str) -> None:
    try:
        hf_hub_download(dataset, filename="README.md", repo_type="dataset", revision="main")
    except HfHubHTTPError as e:
        raise ValueError(f"Dataset '{dataset}' not found or inaccessible.") from e

def push_model(local_dir: str, repo_id: str, token: str | None = None) -> str:
    token = token or os.getenv("HF_TOKEN") 
    for attempt in range(1, 4):
        try:
            _api.create_repo(
                repo_id, 
                exist_ok=True, 
                repo_type="model", 
                token=token)
            upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model", token=token)
            return f"https://huggingface.co/{repo_id}"
        except Exception as e:
            log.warning("HF upload attempt %s failed: %s", attempt, e, exc_info=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to push model to Hugging Face after 3 retries.")
