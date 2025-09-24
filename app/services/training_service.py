import logging, shutil, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from app.schemas.training import TrainingRequest
from storage.job_store import JobStore, JobStatus
from ml.registry import ModelRegistry
from app.core.hf_utils import build_repo_id, ensure_dataset_exists, push_model

try:
    import wandb
except ImportError:                                     
    wandb = None                                        

log = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trainer")


class TrainingService:
    def __init__(self, job_store: JobStore, model_store: object | None = None):
        self.job_store = job_store
        self.model_store = model_store

    def start_job(self, req: TrainingRequest) -> str:
        ensure_dataset_exists(req.dataset)
        model_cls = ModelRegistry.get_model(req.model_type)  # raises KeyError
        job_id = str(uuid.uuid4())

        self.job_store.add(
            job_id=job_id,
            model_type=req.model_type,
            dataset=req.dataset,
            user_id=req.user_id,
            extra={"config": req.config},
        )
        EXECUTOR.submit(self._run_job, job_id, req, model_cls)
        return job_id

    def _run_job(self, job_id: str, req: TrainingRequest, model_cls):
        artefact_dir = Path("storage") / "artifacts" / job_id
        artefact_dir.mkdir(parents=True, exist_ok=True)
        logs: list[Dict[str, str]] = []

        def log_step(msg: str):
            ts = datetime.utcnow().isoformat()
            logs.append({"timestamp": ts, "message": msg})
            self.job_store.update(job_id, logs=logs)

        try:
            log_step("Downloading & splitting dataset...")
            train_path, val_path = self._prepare_dataset(
                req.dataset, artefact_dir, req.config
            )

            log_step(f"Initialising {req.model_type} model...")
            if callable(model_cls):  
                model = model_cls(config=req.config)
            else:
                model = model_cls

            wandb_run = None
            if wandb is not None:
                wandb_run = wandb.init(
                    project="ml-training-pipeline",
                    name=f"{req.user_id}-{job_id[:8]}",
                    config=req.config,
                    reinit=True,
                )
                self.job_store.update(job_id, wandb_run_url=wandb_run.url)

            log_step("Training...")
            # 4. training loop -------------------------------------------------- #
            log_step("Training...")

            import inspect
            import pandas as pd

            # ------------------------------------------------------ #
            # Prepare canonical artefacts once; we’ll reuse as needed
            # ------------------------------------------------------ #
            df_train = pd.read_csv(train_path)
            df_val   = pd.read_csv(val_path)
            target   = req.config.get("target_column", "target")
            feats    = req.config.get("feature_columns") or [c for c in df_train.columns if c != target]

            X_train, y_train = df_train[feats], df_train[target]
            X_val,   y_val   = df_val[feats],   df_val[target]

            # Common kwargs every API style might accept
            common_kw = dict(
                progress_cb=lambda p: self.job_store.update(job_id, progress=p),
                artefact_dir=artefact_dir,
            )

            # ------------------------------------------------------ #
            # Modern models: .fit(train_data_path=…, val_data_path=…)
            # ------------------------------------------------------ #
            if hasattr(model, "fit"):
                metrics = model.fit(
                    train_data_path=train_path,
                    val_data_path=val_path,
                    target_column=target,
                    feature_columns=feats,
                    **common_kw,
                )

            # ------------------------------------------------------ #
            # Legacy models: inspect .train() and pass only what fits
            # ------------------------------------------------------ #
            elif hasattr(model, "train"):
                sig = inspect.signature(model.train)
                want = sig.parameters.keys()

                # Decide which calling convention to use
                if {"X_train", "y_train"}.issubset(want):
                    # four-array style ------------------------------------------------
                    call_kw = {
                        "X_train": X_train,
                        "y_train": y_train,
                        "X_val":   X_val,
                        "y_val":   y_val,
                        **{k: v for k, v in common_kw.items() if k in want},
                    }
                    metrics = model.train(**call_kw)

                else:
                    # single-path style ----------------------------------------------
                    path_kw = next(
                        (k for k in ("dataset_path", "data_path", "csv_path") if k in want),
                        None,
                    )
                    positional = [] if path_kw else [train_path]
                    call_kw = {path_kw: train_path} if path_kw else {}
                    call_kw.update({k: v for k, v in common_kw.items() if k in want})
                    if "target_column" in want:
                        call_kw["target_column"] = target
                    if "feature_columns" in want:
                        call_kw["feature_columns"] = feats
                    metrics = model.train(*positional, **call_kw)

            else:
                raise AttributeError(
                    f"{model.__class__.__name__} has neither .fit() nor .train()"
                )

            if wandb_run:
                wandb_run.log(metrics)
                wandb_run.finish()

            model_path = artefact_dir / "model.joblib"          # or .pkl, same idea
            model.save(model_path)
            log_step(f"Saving artefacts to {model_path} …")

            repo_url = None
            if req.upload_to_hub:
                repo_id = build_repo_id(
                    req.user_id,
                    req.model_type,
                    req.dataset.split("/")[-1],
                )
                log_step(f"Pushing to HF repo {repo_id} ...")
                repo_url = push_model(str(artefact_dir), repo_id)
                log_step(f"Upload successful: {repo_url}")

            self.job_store.complete(
                job_id,
                metrics=metrics,
                huggingface_model_id=repo_url,
            )

        except Exception as e:                               
            log.exception("Job %s failed", job_id)
            self.job_store.fail(job_id, error=str(e))
        finally:
            if Path(artefact_dir).exists():
                shutil.rmtree(artefact_dir, ignore_errors=True)

    def _prepare_dataset(
        self, dataset: str, dst_root: Path, cfg: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """
        Downloads the HF dataset and saves train/val CSVs.
        """
        from datasets import load_dataset, Dataset

        ds_dict = load_dataset(dataset)
        if "train" in ds_dict and "validation" in ds_dict:
            train_ds, val_ds = ds_dict["train"], ds_dict["validation"]
        else:                                               # 🆕  split helper
            full: Dataset = ds_dict[list(ds_dict.keys())[0]]
            val_size = cfg.get("val_split", 0.2)
            train_ds = full.shuffle(seed=42).select(
                range(int((1 - val_size) * len(full)))
            )
            val_ds = full.shuffle(seed=42).select(
                range(int((1 - val_size) * len(full)), len(full))
            )

        train_csv = dst_root / "train.csv"
        val_csv = dst_root / "val.csv"
        train_ds.to_csv(train_csv)
        val_ds.to_csv(val_csv)
        return train_csv, val_csv
