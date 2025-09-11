import os
import logging
from logging.handlers import RotatingFileHandler

os.environ["WANDB_SILENT"] = "true"   # suppress W&B service chatter
os.environ["WANDB_CONSOLE"] = "off"   # don't capture/mirror stdout/stderr

import torch
import wandb
from ..utils.metrics import count_parameters
from ..utils.loss import LightweightLoss
from ..utils.logger import _ensure_dir, _build_logger
from ..models.config import create_ablation_models
from .validator import Validator


class AblationTrainer(Validator):
    """Training class for ablation study models with logging."""

    def __init__(self, device='mps',
                 base_channels=12,
                 dataset_name="CVC-ClinicDB",
                 save_path="./ablation_study",
                 use_wandb: bool = True,
                 per_model_logs: bool = True,
                 log_level: int = logging.INFO):
        self.device = device
        self.base_channels = base_channels
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.criterion = LightweightLoss()
        self.use_wandb = use_wandb

        # Logging config
        self.per_model_logs = per_model_logs
        self.log_level = log_level

    def train_model(self, model, train_loader, val_loader, epochs=50, lr=1e-3,
                    model_name="model"):
        """Train a single ablation model"""

        _ensure_dir(self.save_path)
        logger = _build_logger(
            base_dir=self.save_path,
            model_name=model_name,
            per_model=self.per_model_logs,
            level=self.log_level
        )

        model_save_path = f"{self.save_path}/models"
        _ensure_dir(model_save_path)

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses = []
        val_losses = []
        val_dices = []
        val_ious = []

        best_dice = 0.0
        best_model_state = None

        if self.use_wandb:
            wandb.init(
                project="ablation-study", 
                name=model_name, 
                config={
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "model_name": model_name,
                    "dataset": self.dataset_name,
                    "base_channels": self.base_channels
                },
                tags=[self.dataset_name, "ablation"],
                reinit=True
            )

        logger.info(f"Training {model_name}...")
        logger.info(f"Parameters: {count_parameters(model):,}")
        logger.info(f"Base_channels: {self.base_channels}")
        logger.info("-" * 50)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Periodic progress
                if batch_idx % 20 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            # Validation phase
            val_results = self.validate_model(model, val_loader)
            val_loss = val_results['test_loss']
            val_dice = val_results['avg_dice_score']
            val_iou = val_results['avg_iou']
            val_iou_batch = val_results['iou_batch_avg']
            val_dice_batch = val_results['dice_batch_avg']

            # LR schedule
            scheduler.step()

            # Record metrics
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)

            if self.use_wandb:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_iou_batch": val_iou_batch,
                    "val_dice_batch": val_dice_batch,
                    "lr": scheduler.get_last_lr()[0]
                })

            # Save best state (in-memory)
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_state = model.state_dict()

            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Dice: {val_dice:.4f}, "
                f"Val IoU: {val_iou:.4f}"
            )
            logger.info("-" * 50)

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save checkpoint
        if model_save_path:
            _ensure_dir(model_save_path)
            ckpt_path = os.path.join(model_save_path, f"{model_name}_best.pth")
            torch.save({
                'model_state_dict': best_model_state,
                'model_name': model_name,
                'best_dice': best_dice,
                'config': model.config if hasattr(model, 'config') else None
            }, ckpt_path)
            logger.info(f"Saved best checkpoint to: {ckpt_path}")

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'val_ious': val_ious,
            'best_dice': best_dice,
            'parameters': count_parameters(model)
        }

        if self.use_wandb:
            wandb.finish()

        logger.info(f"Finished training {model_name}. Best Dice: {best_dice:.4f}")
        return model, training_history

    def train_all_ablation_models(self, train_loader, val_loader, epochs=100):
        """Train all ablation study models"""
        _ensure_dir(self.save_path)
        # Aggregate logger for "orchestration" messages
        orchestration_logger = _build_logger(
            base_dir=self.save_path,
            model_name="ALL",
            per_model=self.per_model_logs,  # still creates separate per-model logs inside train_model
            level=self.log_level
        )

        models = create_ablation_models(self.base_channels)
        results = {}

        for model_name, model_info in models.items():
            orchestration_logger.info("=" * 60)
            orchestration_logger.info(f"Training: {model_info['description']}")
            orchestration_logger.info("=" * 60)

            model = model_info['model']
            model.config = model_info['config']  # Add config for saving

            trained_model, history = self.train_model(
                model, train_loader, val_loader, epochs,
                model_name=model_name
            )

            results[model_name] = {
                'model': trained_model,
                'history': history,
                'description': model_info['description'],
                'config': model_info['config']
            }

            orchestration_logger.info(f"Completed: {model_name}")
            orchestration_logger.info(f"Best Dice: {history['best_dice']:.4f}")
            orchestration_logger.info(f"Parameters: {history['parameters']:,}")

        return results
