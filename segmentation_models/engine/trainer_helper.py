import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..utils.callbacks import wb, setup_logger, append_config_block
from ..config import TrainConfig, LOSS, OPTIM


class TrainerHelper:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: TrainConfig = TrainConfig(),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.train_losses, self.val_losses = [], []
        self.val_dices, self.val_ious = [], []
        
        self.cfg = cfg

        self.model_name = cfg.model_name.replace("-", "_")
        self.dataset_name = cfg.dataset_name.replace("-", "_")

        # Paths & timestamp
        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(cfg.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.base_dir / "models"; self.models_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = self.base_dir / "history"; self.history_dir.mkdir(parents=True, exist_ok=True)
        self.loss_dir = self.base_dir / "loss"; self.loss_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_dir = self.base_dir / "visualizations"; self.visualize_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.base_dir / "logs"; self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = str(self.logs_dir)

        # Logger
        self.logger, self.log_path = setup_logger(name=f"{self.model_name}_model_{self.dataset_name}_{self.time_stamp}", logs_dir=self.logs_dir)
        append_config_block(self.log_path, asdict(cfg), title="CONFIG", sort_keys=True)
        self.logger.info("Logging to %s", self.log_path)

        # --- Loss ---
        if self.cfg.criterion not in LOSS:
            raise ValueError(f"Unknown criterion: {self.cfg.criterion}. Choose from {list(LOSS)}")
        self.criterion = LOSS[self.cfg.criterion]()

        # --- Optimizer ---
        if self.cfg.optimizer not in OPTIM:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}. Choose from {list(OPTIM)}")
        self.optimizer = OPTIM[self.cfg.optimizer](
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # Loss
        if self.cfg.criterion not in LOSS:
            raise ValueError(f"Unknown criterion: {self.cfg.criterion}. Choose from {list(LOSS)}")
        self.criterion = LOSS[self.cfg.criterion]()  # no kwargs change from your config

        # Optimizer
        if self.cfg.optimizer not in OPTIM:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}. Choose from {list(OPTIM)}")
        self.optimizer = OPTIM[self.cfg.optimizer](
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        self.scheduler = self._build_scheduler()

        # W&B
        self.wandb_run = None
        if cfg.use_wandb:
            run_name = cfg.wandb_run_name or f"{self.model_name}_model_{self.dataset_name}_{self.time_stamp}"
            
            self.wandb_run = wb.init(cfg.wandb_entity, cfg.wandb_project, run_name, config={
                "optimizer": type(self.optimizer).__name__,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "scheduler": type(self.scheduler).__name__,
                "scheduler_factor": cfg.scheduler_factor,
                "scheduler_patience": cfg.scheduler_patience,
                "epochs": cfg.epochs,
                "early_stopping_patience": cfg.early_stopping_patience,
                "criterion": type(self.criterion).__name__,
                "threshold": cfg.threshold,
            }, reinit=True)
            if self.wandb_run is not None:
                self.logger.info("W&B URL: %s", getattr(self.wandb_run, "url", ""))
    
    # ---- helpers ----
    def _save_best(self, epoch: int, val_loss: float) -> str:
        p = self.models_dir / f"best_{self.model_name}_model_{self.dataset_name}_{self.time_stamp}.pth"
        torch.save(self.model.state_dict(), p)
        self.logger.info("Saved best: %s (val=%.6f)", p, val_loss)
        if self.wandb_run is not None:
            wb.log_artifact(
                self.wandb_run,
                file_path=str(p),
                name=f"best_{self.model_name}_model_{self.time_stamp}",
                type="model",
                metadata={"val_loss": val_loss, "epoch": epoch},
            )
        return str(p)

    def _save_history(self) -> str:
        df = pd.DataFrame({"epoch": list(range(1, len(self.train_losses)+1)), "train_loss": self.train_losses, "val_loss": self.val_losses})
        p = self.history_dir / f"loss_history_{self.model_name}_{self.dataset_name}_{self.time_stamp}.csv"
        df.to_csv(p, index=False)
        self.logger.info("Saved history: %s", p)
        if self.wandb_run is not None:
            wb.log_artifact(self.wandb_run, file_path=str(p), name=f"loss_history_{self.model_name}_model_{self.dataset_name}_{self.time_stamp}", type="metrics")
        return str(p)
    
    def _build_scheduler(self):
        # Default: Plateau on val loss
        if self.cfg.scheduler is None:
            return None
        
        name = self.cfg.scheduler

        if self.cfg.scheduler == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.cfg.mode,
                factor=self.cfg.scheduler_factor,
                patience=self.cfg.scheduler_patience,
            )

        if name == "CosineAnnealingLR":
            # T_max defaults to num epochs unless provided
            T_max = getattr(self.cfg, "T_max", self.cfg.epochs)
            eta_min = getattr(self.cfg, "eta_min", 0.0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)

        if name == "CosineAnnealingWarmRestarts":
            # Reasonable defaults if not provided
            T_0 = getattr(self.cfg, "T_0", max(1, self.cfg.epochs // 5))
            T_mult = getattr(self.cfg, "T_mult", 2)
            eta_min = getattr(self.cfg, "eta_min", 0.0)
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        if name == "StepLR":
            step_size = getattr(self.cfg, "step_size", max(1, self.cfg.epochs // 3))
            gamma = getattr(self.cfg, "gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        if name == "MultiStepLR":
            milestones = getattr(
                self.cfg, "milestones",
                [int(self.cfg.epochs * 0.5), int(self.cfg.epochs * 0.75)]
            )
            gamma = getattr(self.cfg, "gamma", 0.1)
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        if name == "ExponentialLR":
            gamma = getattr(self.cfg, "gamma", 0.95)
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        if name == "OneCycleLR":
            # Needs per-iteration stepping; we set it up here
            steps_per_epoch = len(self.train_loader)
            max_lr = getattr(self.cfg, "max_lr", self.cfg.lr)
            pct_start = getattr(self.cfg, "pct_start", 0.3)
            div_factor = getattr(self.cfg, "div_factor", 25.0)
            final_div_factor = getattr(self.cfg, "final_div_factor", 1e4)
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=self.cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )

        if name == "CyclicLR":
            # Also per-iteration stepping
            base_lr = getattr(self.cfg, "base_lr", self.cfg.lr / 10)
            max_lr = getattr(self.cfg, "max_lr", self.cfg.lr)
            mode = getattr(self.cfg, "cyclic_mode", "triangular2")
            step_size_up = getattr(self.cfg, "step_size_up", max(1, len(self.train_loader) // 2))
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                mode=mode,
                step_size_up=step_size_up,
                cycle_momentum=False,  # often False when not using SGD with momentum
            )

        raise ValueError(f"Unknown scheduler: {name}")

    def plot_losses(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(self.loss_dir / f"loss_plot_{self.model_name}_model_{self.dataset_name}_{self.time_stamp}.png")
        plt.show()

    def visualize_output_samples(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Convert test_loader into a list of batches (not memory-heavy if dataset is small/medium)
        batches = list(self.test_loader)

        # Pick a random batch
        batch_idx = random.randint(0, len(batches) - 1)
        images, masks = batches[batch_idx]

        images, masks = images.to(self.device), masks.to(self.device)

        # Number of samples to visualize
        num_examples = 3
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 12))

        for i in range(num_examples):
            # Pick a random sample from this batch
            idx = random.randint(0, images.size(0) - 1)

            image = images[idx].unsqueeze(0)  # add batch dimension
            mask = masks[idx]

            # Forward pass
            with torch.no_grad():
                output = self.model(image)
                output = torch.sigmoid(output).cpu().numpy().squeeze()
                output = (output > 0.5)  # binary mask

            # Plot original, GT mask, and prediction
            axes[i, 0].imshow(images[idx].cpu().permute(1, 2, 0))
            axes[i, 0].set_title("Original Image")

            axes[i, 1].imshow(mask.cpu().squeeze(), cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")

            axes[i, 2].imshow(output, cmap="gray")
            axes[i, 2].set_title("Model Output Mask")

        plt.tight_layout()
        plt.savefig(self.visualize_dir / f"test_results_{self.model_name}_model_{self.dataset_name}_{self.time_stamp}.png")
        plt.show()
