import torch
from tqdm import tqdm
import numpy as np

from ..utils import *


class Validator:
    def validate(self, loader=None, input_size=(3, 256, 256)):
        if loader is None:
            loader = self.test_loader
        self.model.eval()

        # Calculate model parameters
        total_params, trainable_params = count_parameters(self.model)
        flops, gflops = calculate_flops(self.model, input_size, self.device)

        avg_inference_time = measure_inference_time(self.model, loader, self.device)
        fps = 1.0 / avg_inference_time
        
        test_loss = 0.0
        all_labels = [] # Store all ground truth masks
        all_preds = [] # Store all predicted masks
        all_iou_scores = [] # Store IoU scores for each batch
        all_dice_scores = [] # Store Dice scores for each batch
        all_precision_scores = [] # Store per-image precision scores
        all_recall_scores = [] # Store per-image recall scores
        all_f1_scores = [] # Store per-image F1 scores (macro-averaged, matches Dice convention)
        all_hd_scores = []
        dice_total = 0.0
        iou_total = 0.0

        with torch.no_grad():
            for images, masks in tqdm(loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                # Handle auxiliary outputs
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                loss = self.criterion(outputs, masks)
                test_loss += loss.item()

                outputs = (torch.sigmoid(main_output) > 0.5).float() # Convert to binary mask
                outputs_binary = outputs.squeeze(1) # Remove extra channel dimension
                masks = masks.squeeze(1) # Ensure ground truth also has correct dimensions
                
                iou_scores = calculate_iou(outputs_binary, masks) # Use the calculate_iou function
                dice_scores = calculate_dice_score(outputs_binary, masks)
                precision_scores, recall_scores, f1_scores = calculate_precision_recall_f1(outputs_binary, masks)
                hd_scores = calculate_hausdorff_distance(outputs_binary, masks) # Use the calculate_hausdorff_distance function

                all_dice_scores.extend(dice_scores.cpu().numpy().tolist()) # Store Dice scores
                all_precision_scores.extend(precision_scores.cpu().numpy().tolist())
                all_recall_scores.extend(recall_scores.cpu().numpy().tolist())
                all_f1_scores.extend(f1_scores.cpu().numpy().tolist())
                all_hd_scores.extend(hd_scores.cpu().numpy().tolist()) # Store Hausdorff distances
                all_iou_scores.extend(iou_scores.cpu().numpy().tolist()) # Store IoU values
                all_labels.extend(masks.cpu().numpy().tolist())
                all_preds.extend(outputs_binary.cpu().numpy().tolist())

                dice = dice_score(outputs_binary, masks)
                iou = iou_score(outputs_binary, masks)
                dice_total += dice.item()
                iou_total += iou.item()

        # Pixel-pooled (micro-averaged) counterparts, kept only as a secondary
        # diagnostic. Reported headline metrics (precision/recall/f1 below) use
        # the same per-image macro-averaging as Dice/IoU, per the methodology
        # stated in the manuscript -- this is what makes Dice and F1 identical
        # by construction, resolving the earlier Table 3 inconsistency.
        all_preds_np = np.array(all_preds).astype(np.uint8).flatten()
        all_labels_np = np.array(all_labels).astype(np.uint8).flatten()
        tp_pool = int(np.sum((all_preds_np == 1) & (all_labels_np == 1)))
        fp_pool = int(np.sum((all_preds_np == 1) & (all_labels_np == 0)))
        fn_pool = int(np.sum((all_preds_np == 0) & (all_labels_np == 1)))
        precision_micro = tp_pool / (tp_pool + fp_pool + 1e-7)
        recall_micro = tp_pool / (tp_pool + fn_pool + 1e-7)
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-7)

        avg_test_loss = test_loss / len(loader)
        avg_dice_score = sum(all_dice_scores) / len(all_dice_scores) if all_dice_scores else 0 # Calculate average Dice score
        avg_hd_score = sum(all_hd_scores) / len(all_hd_scores) if all_hd_scores else 0 # Calculate average Hausdorff distance
        avg_iou = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0 # Calculate average IoU
        precision = sum(all_precision_scores) / len(all_precision_scores) if all_precision_scores else 0
        recall = sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0
        f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

        # Prepare results dictionary
        results = {
            'test_loss': avg_test_loss,
            'avg_dice_score': avg_dice_score,
            'avg_hd_score': avg_hd_score,
            'avg_iou': avg_iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'iou_batch_avg': iou_total / len(loader),
            'dice_batch_avg': dice_total / len(loader),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'flops': flops,
            'gflops': gflops
        }

        return results
    
    def validate_and_save(self, input_size=(3, 256, 256)):
        results = self.validate(loader=self.test_loader, input_size=input_size)
        csv_path = f"{self.cfg.model_name}_model_{self.cfg.dataset_name}_evaluation_results"
        folder_name = self.base_dir / "evaluation_results"
        folder_name = str(folder_name)
        save_results_to_csv(results, folder_name=folder_name, filename=csv_path, timestamp=self.time_stamp)
        print(f"Validation results saved to {folder_name}/{csv_path}")
        return results
