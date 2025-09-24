import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import requests

BASE_URL: str = os.getenv("GENEALPHA_API_URL", "http://localhost:8000/api/v1")
API_PREFIX: str = "/training"       
LOG_FILE = Path(__file__).with_suffix(".log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, "w", "utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("e2e")

def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)

def _req(method: str, path: str, **kw) -> requests.Response:
    url = f"{BASE_URL}{path}"
    show = f"{method} {url}"
    if kw.get("params"):
        show += f"  params={kw['params']}"
    log.info("► %s", show)
    if kw.get("json"):
        log.info("│ Request JSON:\n%s", _pretty(kw["json"]))
    r = requests.request(method, url, timeout=30, **kw)
    log.info("◄ %s %s", r.status_code, r.reason)
    try:
        log.info("│ Response JSON:\n%s", _pretty(r.json()))
    except ValueError:
        log.info("│ Non-JSON body: %s", r.text[:200])
    r.raise_for_status()
    return r

def list_datasets() -> List[str]:
    resp = _req("GET", "/datasets")
    return resp.json()["datasets"]

def list_models() -> List[str]:
    resp = _req("GET", f"{API_PREFIX}/models")
    return resp.json()["models"]

def start_training(
    user_id: str,
    model_type: str,
    dataset: str,
) -> str:
    payload = {
        "user_id": user_id,
        "model_type": model_type,
        "dataset": dataset,
        "config": {
            "target_column": "target",
        },
        "upload_to_hub": True,
    }
    resp = _req("POST", f"{API_PREFIX}/start", json=payload)
    return resp.json()["job_id"]

def poll_job(job_id: str, every: int = 5, timeout_min: int = 10) -> Dict[str, Any]:
    path = f"/jobs/{job_id}"
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        resp = _req("GET", path)
        data = resp.json()
        status, progress = data["status"].lower(), data["progress"]
        log.info("Job %s – %s (%s%%)", job_id, status, progress)
        if status in ("completed", "failed"):
            return data
        time.sleep(every)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_min} min")

def assert_repo_pattern(repo_url: str, user: str, model: str, dataset: str) -> None:
    """
    GeneAlpha/<user>_<model>_<dataset>__YYYYMMDD-HHMMSS-xxxx
    """
    patt = (
        rf"^https://huggingface\.co/"
        rf"GeneAlpha/{re.escape(user)}_{re.escape(model)}_{re.escape(dataset)}"
        rf"__\d{{8}}-\d{{6}}-[0-9a-f]{{6}}$"
    )
    assert re.match(patt, repo_url), f"Repo url {repo_url} does not match {patt}"

def negative_invalid_model(valid_dataset: str):
    payload = {
        "user_id": "negTester",
        "model_type": "does_not_exist",
        "dataset": valid_dataset,
        "config": {},
    }
    try:
        _req("POST", f"{API_PREFIX}/start", json=payload)
    except requests.HTTPError as e:
        assert e.response.status_code == 400, "Expected 400 for invalid model"
        log.info("Negative test ✅  invalid model rejected.")
    else:
        raise AssertionError("Invalid model was accepted!")

def negative_invalid_dataset(valid_model: str):
    payload = {
        "user_id": "negTester",
        "model_type": valid_model,
        "dataset": "GeneAlpha/this_dataset_is_fake",
        "config": {},
    }
    try:
        _req("POST", f"{API_PREFIX}/start", json=payload)
    except requests.HTTPError as e:
        assert e.response.status_code == 400, "Expected 400 for invalid dataset"
        log.info("Negative test ✅  invalid dataset rejected.")
    else:
        raise AssertionError("Invalid dataset was accepted!")

def main():
    log.info("==============  GeneAlpha E2E user-flow ==============")

    # sanity endpoints
    datasets = list_datasets()
    models = list_models()
    if not (datasets and models):
        raise RuntimeError("Datasets or models list empty!")

    user_id = "alice42"
    dataset_key = datasets[0].split("/")[-1]
    model_key = "linear_regression" if "linear_regression" in models else models[0]

    # 1. happy-path start + poll
    job_id = start_training(user_id, model_key, datasets[0])
    job_info = poll_job(job_id)
    assert job_info["status"] == "completed", f"Job failed: {job_info.get('error')}"
    repo_url = job_info["huggingface_model_id"]
    assert_repo_pattern(repo_url, user_id, model_key, dataset_key)
    log.info("Happy path ✅  repo url matches pattern.")

    # 2. negative cases
    negative_invalid_model(datasets[0])
    negative_invalid_dataset(model_key)

    log.info("✓ All E2E checks passed. Detailed log in %s", LOG_FILE)

# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
