
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, Tuple

def calculate_iou(outputs, masks):
    """
    Calculates the Intersection over Union (IoU) for segmentation results.

    Args:
        outputs (torch.Tensor): Model outputs (binary masks), shape [batch_size, height, width].
        masks (torch.Tensor): Ground truth masks, shape [batch_size, height, width].

    Returns:
        torch.Tensor: IoU scores, shape [batch_size].
    """
    with torch.no_grad():
        outputs = outputs.long()  # Convert to long type
        masks = masks.long()

        intersection = (outputs & masks).sum(dim=(1, 2)).float()  # Element-wise AND for intersection
        union = (outputs | masks).sum(dim=(1, 2)).float()  # Element-wise OR for union

        iou = (intersection + 1e-7) / (union + 1e-7)  # Avoid division by zero

    return iou

def calculate_dice_score(outputs, masks, threshold=0.5):
    """
    Calculates the Dice Similarity Coefficient (DSC) for segmentation results.

    Args:
        outputs (torch.Tensor): Model outputs (binary masks), shape [batch_size, height, width].
        masks (torch.Tensor): Ground truth masks, shape [batch_size, height, width].

    Returns:
        torch.Tensor: Dice scores, shape [batch_size].
    """

    with torch.no_grad():
        outputs = outputs.long()  # Convert to long type
        masks = masks.long()

        intersection = (outputs & masks).sum(dim=(1, 2)).float()  # Element-wise AND for intersection
        dice_score = (2 * intersection + 1e-7) / (outputs.sum(dim=(1, 2)).float() + masks.sum(dim=(1, 2)).float() + 1e-7)  # Avoid division by zero

    return dice_score

def calculate_precision_recall_f1(outputs, masks, eps=1e-7):
    """
    Calculates per-image precision, recall, and F1 score, matching the
    per-image macro-averaging convention used by calculate_dice_score/calculate_iou.

    Args:
        outputs (torch.Tensor): Binary predicted masks, shape [batch_size, height, width].
        masks (torch.Tensor): Ground truth masks, shape [batch_size, height, width].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: precision, recall, f1 per image, shape [batch_size].
    """
    with torch.no_grad():
        outputs = outputs.long()
        masks = masks.long()

        tp = (outputs & masks).sum(dim=(1, 2)).float()
        fp = (outputs & (1 - masks)).sum(dim=(1, 2)).float()
        fn = ((1 - outputs) & masks).sum(dim=(1, 2)).float()

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

    return precision, recall, f1

def calculate_hausdorff_distance(outputs, masks, spacing=(1.0, 1.0)):
    """
    Calculates the Average Hausdorff Distance (HD) in mm for segmentation results.

    Args:
        outputs (torch.Tensor): Model outputs (binary masks), shape [batch_size, height, width].
        masks (torch.Tensor): Ground truth masks, shape [batch_size, height, width].
        spacing (tuple): Pixel spacing in mm (height_spacing, width_spacing).

    Returns:
        torch.Tensor: Average Hausdorff Distance for each sample in the batch, shape [batch_size].
    """

    with torch.no_grad():
        outputs = outputs.cpu().numpy().astype(np.bool_)
        masks = masks.cpu().numpy().astype(np.bool_)

        batch_hd = []
        for output, mask in zip(outputs, masks):
            output_points = np.argwhere(output) * spacing
            mask_points = np.argwhere(mask) * spacing

            if output_points.size == 0 or mask_points.size == 0:
                batch_hd.append(float('inf'))  # Handle empty masks
                continue

            hd1 = directed_hausdorff(output_points, mask_points)[0]
            hd2 = directed_hausdorff(mask_points, output_points)[0]
            batch_hd.append(max(hd1, hd2))

        return torch.tensor(batch_hd, dtype=torch.float32)

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2 * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)


def _binary_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Tuple[int,int,int,int]:
    """Return TP, FP, FN, TN for binary segmentation from raw logits.
    Shapes accepted: logits (B,1,H,W) or (B,H,W); targets (B,1,H,W)/(B,H,W)."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    probs = torch.sigmoid(logits)
    preds = probs > threshold

    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    targets = targets > 0.5

    preds = preds.detach().to(torch.bool).cpu().view(-1)
    targets = targets.detach().to(torch.bool).cpu().view(-1)

    tp = torch.sum(preds & targets).item()
    fp = torch.sum(preds & (~targets)).item()
    fn = torch.sum((~preds) & targets).item()
    tn = torch.sum((~preds) & (~targets)).item()
    return int(tp), int(fp), int(fn), int(tn)


def _multiclass_counts(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int,int,int,int]:
    """Return micro-averaged TP, FP, FN, TN across classes from raw logits.
    logits: (B,C,H,W); targets: (B,H,W) or (B,1,H,W) or one-hot (B,C,H,W)."""
    C = logits.shape[1]
    preds = torch.argmax(logits, dim=1).detach().cpu().view(-1)

    if targets.ndim == 4 and targets.shape[1] > 1:
        targets = torch.argmax(targets, dim=1)
    elif targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    targets = targets.detach().cpu().long().view(-1)

    tp = fp = fn = 0
    for c in range(C):
        pc = preds == c
        tc = targets == c
        tp += torch.sum(pc & tc).item()
        fp += torch.sum(pc & (~tc)).item()
        fn += torch.sum((~pc) & tc).item()
    total = preds.numel()
    tn = total - (tp + fp + fn)
    return int(tp), int(fp), int(fn), int(tn)


def update_counts_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[int,int,int,int]:
    """Detect binary vs multiclass by channel dim and return (TP, FP, FN, TN)."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        return _binary_counts(logits, targets, threshold)
    else:
        return _multiclass_counts(logits, targets)


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-7) -> Dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }

