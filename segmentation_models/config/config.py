from dataclasses import dataclass
from typing import Optional
from torch import nn, optim

from .. import DiceLoss, BCEDiceLoss, ResUNetDeepSupervisionLoss, LightweightLoss

LOSS = {
    # Binary segmentation (single foreground vs background)
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,   # expects logits shape [B,1,H,W] or [B,H,W]
    "DiceLoss": DiceLoss,
    "BCEDiceLoss": BCEDiceLoss,
    "ResUNetDeepSupervisionLoss": ResUNetDeepSupervisionLoss,
    "LightweightLoss": LightweightLoss,

    # Multiclass semantic segmentation (mutually exclusive classes)
    "CrossEntropyLoss": nn.CrossEntropyLoss,     # logits [B,C,H,W], targets Long [B,H,W], supports ignore_index
}

OPTIM = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,        # often with momentum for large-scale training
    "RMSprop": optim.RMSprop
}


@dataclass
class TrainConfig:
    # --- General ---
    model_name: str = "UNet"
    dataset_name: str = "CVC-ClinicDB"
    criterion: str = "BCEDiceLoss"
    optimizer: str = "AdamW"
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # --- Scheduler ---
    scheduler: Optional[str] = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5          # for ReduceLROnPlateau
    scheduler_patience: int = 5            # for ReduceLROnPlateau
    mode: str = "min"

    # Shared / optional scheduler args
    gamma: float = 0.1                      # StepLR, MultiStepLR, ExponentialLR (default 0.1)
    step_size: int = 1                      # StepLR (default 1)
    milestones: Optional[list[int]] = None  # MultiStepLR (default empty list)
    T_max: int = 100                        # CosineAnnealingLR (default 100)
    T_0: int = 1                            # CosineAnnealingWarmRestarts (default 1)
    T_mult: int = 2                         # CosineAnnealingWarmRestarts (default 2)
    eta_min: float = 0.0                    # Cosine schedulers (default 0.0)

    # OneCycleLR / CyclicLR
    max_lr: float = 0.1                     # OneCycleLR (must be set, use 0.1 as safe default)
    base_lr: float = 0.001                  # CyclicLR (default base LR)
    pct_start: float = 0.3                  # OneCycleLR (default 0.3)
    div_factor: float = 25.0                # OneCycleLR (default 25.0)
    final_div_factor: float = 1e4           # OneCycleLR (default 1e4)
    step_size_up: int = 2000                # CyclicLR (default 2000 steps up)
    cyclic_mode: str = "triangular2"        # CyclicLR (default triangular2)

    # --- Training control ---
    early_stopping_patience: int = 10
    gradient_clipping: Optional[float] = None

    # --- Paths / logging ---
    base_dir: str = "model_comparisons"
    models_dir: str = "trained_models"
    history_dir: str = "models_history"
    loss_dir: str = "loss_plots"
    visualize_dir: str = "visualizations"
    logs_dir: str = "logs"
    use_wandb: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = "segmentation_models"
    wandb_run_name: Optional[str] = None
    wandb_watch_log_freq: int = 100

    # --- Metrics ---
    threshold: float = 0.5
    log_train_metrics: bool = True
