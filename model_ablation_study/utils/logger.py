
import os
import time
import logging
from logging.handlers import RotatingFileHandler

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _build_logger(base_dir: str,
                  model_name: str,
                  aggregate_filename: str = "ablation_all.log",
                  per_model: bool = True,
                  level: int = logging.INFO,
                  max_bytes: int = 10 * 1024 * 1024,
                  backup_count: int = 5) -> logging.Logger:
    """
    Create (or reuse) a logger that writes to:
      - aggregate file (shared by all models)
      - optional per-model file
    """
    logs_dir = _ensure_dir(os.path.join(base_dir, "logs"))
    ts = time.strftime("%Y%m%d-%H%M%S")

    logger_name = f"AblationTrainer.{model_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs in root

    # Prevent adding duplicate handlers if called again
    existing_paths = {getattr(h, "baseFilename", None) for h in logger.handlers}

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Aggregate handler
    agg_path = os.path.join(logs_dir, aggregate_filename)
    if agg_path not in existing_paths:
        agg_handler = RotatingFileHandler(agg_path, maxBytes=max_bytes, backupCount=backup_count)
        agg_handler.setLevel(level)
        agg_handler.setFormatter(fmt)
        logger.addHandler(agg_handler)

    # Per-model handler
    if per_model:
        per_model_path = os.path.join(logs_dir, f"{model_name}.log")
        if per_model_path not in existing_paths:
            model_handler = RotatingFileHandler(per_model_path, maxBytes=max_bytes, backupCount=backup_count)
            model_handler.setLevel(level)
            model_handler.setFormatter(fmt)
            logger.addHandler(model_handler)

    # Optional: stream to console (comment out if you don’t want console logs)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger

