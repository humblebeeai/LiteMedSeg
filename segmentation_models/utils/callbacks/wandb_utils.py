# wandb_utils.py
from typing import Any, Dict, Optional


from typing import Optional, Dict, Any

def init(
    entity: Optional[str] = None,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    reinit: bool = True,
):
    try:
        import wandb  # type: ignore
    except Exception:
        return None

    # Build kwargs dynamically so we don't pass None values
    kwargs = {
        "name": run_name,
        "config": config or {},
        "reinit": reinit,
        "settings": wandb.Settings(save_code=False),
    }
    if entity is not None:
        kwargs["entity"] = entity
    if project is not None:
        kwargs["project"] = project

    return wandb.init(**kwargs)

def log(run, data: Dict[str, Any]) -> None:
    if run is None:
        return
    import wandb  # type: ignore
    wandb.log(data)

def log_artifact(run, file_path: str, name: str, type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    if run is None:
        return
    import wandb  # type: ignore
    art = wandb.Artifact(name=name, type=type, metadata=metadata or {})
    art.add_file(str(file_path))
    run.log_artifact(art)

def finish(run) -> None:
    if run is None:
        return
    import wandb  # type: ignore
    wandb.finish()
