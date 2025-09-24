from fastapi import APIRouter
from functools import lru_cache
from huggingface_hub import HfApi

router = APIRouter(prefix="/datasets", tags=["Datasets"])
_hf = HfApi()

@lru_cache(maxsize=1)
def _fetch_org_datasets():
    return sorted(ds.id for ds in _hf.list_datasets(author="GeneAlpha"))

@router.get("/", summary="List available datasets")
async def list_datasets():
    return {"datasets": _fetch_org_datasets(), "total": len(_fetch_org_datasets())}
