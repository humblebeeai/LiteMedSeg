import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from ..utils.metrics import (
    calculate_dice_score, 
    calculate_iou, 
    calculate_hausdorff_distance,
    count_parameters,
    measure_inference_time,
    calculate_flops,
    dice_score,
    iou_score
)

class Validator:
    """Validation class for ablation study models"""
    def validate_model(self, model, val_loader):
        """Validate a model"""

        model.eval()

        input_size = (3, 256, 256)  # Example: RGB image 256x256, modify as needed
        flops, gflops = calculate_flops(model, input_size, self.device)
        avg_inference_time = measure_inference_time(model, val_loader, self.device)
        fps = 1.0 / avg_inference_time

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        all_labels = [] # Store all ground truth masks
        all_preds = [] # Store all predicted masks
        all_iou_scores = [] # Store IoU scores for each batch
        all_dice_scores = [] # Store Dice scores for each batch
        all_hd_scores = [] # Store Hausdorff distance scores for each batch

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                
                # Handle auxiliary outputs
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                outputs = (torch.sigmoid(main_output) > 0.5).float()
                outputs_binary = outputs.squeeze(1)
                masks = masks.squeeze(1)

                iou_scores = calculate_iou(outputs_binary, masks) # Use the calculate_iou function
                dice_scores = calculate_dice_score(outputs_binary, masks)
                hd_scores = calculate_hausdorff_distance(outputs_binary, masks) # Use the calculate_hausdorff_distance function
                
                all_dice_scores.extend(dice_scores.cpu().numpy().tolist()) # Store Dice scores
                all_hd_scores.extend(hd_scores.cpu().numpy().tolist()) # Store Hausdorff distances
                all_iou_scores.extend(iou_scores.cpu().numpy().tolist()) # Store IoU values
                all_labels.extend(masks.cpu().numpy().tolist())
                all_preds.extend(outputs_binary.cpu().numpy().tolist())
                
                dice = dice_score(outputs_binary, masks)
                iou = iou_score(outputs_binary, masks)
                
                total_dice += dice.item()
                total_iou += iou.item()

        all_preds_np = np.array(all_preds).astype(np.uint8).flatten()
        all_labels_np = np.array(all_labels).astype(np.uint8).flatten()

        avg_dice_score = sum(all_dice_scores) / len(all_dice_scores) if all_dice_scores else 0 # Calculate average Dice score
        avg_hd_score = sum(all_hd_scores) / len(all_hd_scores) if all_hd_scores else 0 # Calculate average Hausdorff distance
        avg_iou = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0 # Calculate average IoU
        precision = precision_score(all_labels_np, all_preds_np)
        recall = recall_score(all_labels_np, all_preds_np)
        f1 = f1_score(all_labels_np, all_preds_np)

        n_batches = len(val_loader)
        results = {
            'test_loss': total_loss / n_batches,
            'avg_dice_score': avg_dice_score,
            'avg_hd_score': avg_hd_score,
            'avg_iou': avg_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou_batch_avg': total_iou / n_batches,
            'dice_batch_avg': total_dice / n_batches,
            'total_params': count_parameters(model),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'flops': flops,
            'gflops': gflops
        }

        return results