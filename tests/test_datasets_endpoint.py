import os, json, logging
from pathlib import Path
from typing import Any, Dict

import requests

BASE_URL = os.getenv("GENEALPHA_API_URL", "http://localhost:8000/api/v1")
LOG_FILE = Path(__file__).with_suffix(".log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, "w", "utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("datasets.e2e")

def _j(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, default=str)

def test_datasets_e2e():
    url = f"{BASE_URL}/datasets"
    log.info("► GET %s", url)
    resp = requests.get(url, timeout=20)
    log.info("◄ %s %s", resp.status_code, resp.reason)
    log.info("│ Body:\n%s", _j(resp.json()))

    assert resp.status_code == 200, resp.text
    data = resp.json()

    print("\nDatasets returned by the API:")
    for i, ds in enumerate(data["datasets"], 1):
        print(f"{i:02d}. {ds}")

    # quick validations
    assert data["total"] == len(data["datasets"])
    assert all(ds.startswith("GeneAlpha/") for ds in data["datasets"])

if __name__ == "__main__":
    test_datasets_e2e()
    print(f"✓ /datasets endpoint OK  (log → {LOG_FILE})")
