import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
from typing import Dict, Any

from .. import wb
from .. import update_counts_from_logits, compute_metrics_from_counts
from .. import TrainConfig
from .trainer_helper import TrainerHelper
from .validator import Validator


class SegmentationTrainer(TrainerHelper, Validator):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: TrainConfig = TrainConfig(),
    ) -> None:
        super().__init__(model, device, train_loader, val_loader, test_loader, cfg)
    
    def train(self) -> Dict[str, Any]:
        best_val = float('inf'); es_count = 0
        self.train_losses, self.val_losses = [], []
        best_model_path = None
        start_time = time.time()
        self.val_dices, self.val_ious = [], []
        try:
            for epoch in range(self.cfg.epochs):
                t0 = time.time()
                # ---- Train ----
                self.model.train(); running_train = 0.0
                tp_tr = fp_tr = fn_tr = tn_tr = 0
                for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs} [train]"):
                    images, masks = images.to(self.device), masks.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                    loss.backward()
                    if self.cfg.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clipping)
                    self.optimizer.step()

                    if isinstance(self.scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                        self.scheduler.step()
                    
                    running_train += float(loss.item())

                    if self.cfg.log_train_metrics:
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        tp, fp, fn, tn = update_counts_from_logits(logits.detach(), masks, threshold=self.cfg.threshold)
                        tp_tr += tp; fp_tr += fp; fn_tr += fn; tn_tr += tn

                avg_train = running_train / max(1, len(self.train_loader))
                self.train_losses.append(avg_train)
                train_metrics = compute_metrics_from_counts(tp_tr, fp_tr, fn_tr, tn_tr) if self.cfg.log_train_metrics else None

                results = self.validate(loader=self.val_loader, input_size=(3, 256, 256))

                self.val_losses.append(results["test_loss"])
                self.val_dices.append(results["avg_dice_score"])
                self.val_ious.append(results["avg_iou"])

                # Scheduler & logs
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(results["test_loss"])
                elif self.scheduler is not None and not isinstance(self.scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                    # Epoch-stepped schedulers (Cosine, StepLR, etc.)
                    self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                et = time.time() - t0

                # Console/file
                self.logger.info(
                    "Epoch %03d/%d | Train: %.6f | Val: %.6f | LR: %.6f | %.2fs | Val Dice: %.4f IoU: %.4f F1: %.4f",
                    epoch + 1, self.cfg.epochs, avg_train, results["test_loss"], lr, et, results["avg_dice_score"], results["avg_iou"], results["f1"],
                )

                # W&B
                if self.wandb_run is not None:
                    log_data = {
                        "train/loss": avg_train,
                        "val/loss": results["test_loss"],
                        "lr": lr,
                        "avg_inference_time": results["avg_inference_time"],
                        "avg_fps": results["fps"],
                        "avg_flops": results["flops"],
                        "avg_gflops": results["gflops"],
                        "epoch_seconds": et,
                        "num_trainable": results["trainable_params"],
                        "val/precision": results["precision"],
                        "val/recall": results["recall"],
                        "val/f1": results["f1"],
                        "val/iou": results["avg_iou"],
                        "val/dice": results["avg_dice_score"],
                        "val/iou_batch_avg": results["iou_batch_avg"],
                        "val/dice_batch_avg": results["dice_batch_avg"]
                    }
                    if train_metrics is not None:
                        log_data.update({
                            "train/precision": train_metrics["precision"],
                            "train/recall": train_metrics["recall"],
                            "train/f1": train_metrics["f1"],
                            "train/iou": train_metrics["iou"],
                            "train/dice": train_metrics["dice"]
                        })
                    wb.log(self.wandb_run, log_data)

                # Early stopping
                if results["test_loss"] < best_val:
                    best_val = results["test_loss"]; es_count = 0
                    best_model_path = self._save_best(epoch + 1, best_val)
                else:
                    es_count += 1
                    self.logger.info("No improvement. ES %d/%d", es_count, self.cfg.early_stopping_patience)
                    if es_count >= self.cfg.early_stopping_patience:
                        self.logger.info("Early stopping at epoch %d", epoch + 1)
                        break

            loss_csv = self._save_history()
            avg_epoch = (time.time() - start_time) / max(1, len(self.train_losses))
            self.logger.info("Done. Avg epoch time: %.2fs | Best val: %.6f", avg_epoch, best_val)
            if self.wandb_run is not None:
                try:
                    self.wandb_run.summary["best_val_loss"] = best_val
                    self.wandb_run.summary["epochs_ran"] = len(self.train_losses)
                except Exception:
                    pass
            return {
                "best_val_loss": best_val,
                "epochs_ran": len(self.train_losses),
                "best_model_path": best_model_path,
                "loss_csv_path": loss_csv,
                "wandb_url": getattr(self.wandb_run, "url", None) if self.wandb_run is not None else None,
                "log_path": self.log_path,
            }
        finally:
            wb.finish(self.wandb_run)


def train_segmentation(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, Any]:
    return SegmentationTrainer(model, device, train_loader, val_loader, test_loader, cfg).train()
