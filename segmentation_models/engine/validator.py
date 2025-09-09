import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
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
                hd_scores = calculate_hausdorff_distance(outputs_binary, masks) # Use the calculate_hausdorff_distance function
                
                all_dice_scores.extend(dice_scores.cpu().numpy().tolist()) # Store Dice scores
                all_hd_scores.extend(hd_scores.cpu().numpy().tolist()) # Store Hausdorff distances
                all_iou_scores.extend(iou_scores.cpu().numpy().tolist()) # Store IoU values
                all_labels.extend(masks.cpu().numpy().tolist())
                all_preds.extend(outputs_binary.cpu().numpy().tolist())
                
                dice = dice_score(outputs_binary, masks)
                iou = iou_score(outputs_binary, masks)
                dice_total += dice.item()
                iou_total += iou.item()

        all_preds_np = np.array(all_preds).astype(np.uint8).flatten()
        all_labels_np = np.array(all_labels).astype(np.uint8).flatten()

        avg_test_loss = test_loss / len(loader)
        avg_dice_score = sum(all_dice_scores) / len(all_dice_scores) if all_dice_scores else 0 # Calculate average Dice score
        avg_hd_score = sum(all_hd_scores) / len(all_hd_scores) if all_hd_scores else 0 # Calculate average Hausdorff distance
        avg_iou = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0 # Calculate average IoU
        precision = precision_score(all_labels_np, all_preds_np)
        recall = recall_score(all_labels_np, all_preds_np)
        f1 = f1_score(all_labels_np, all_preds_np)

        # Prepare results dictionary
        results = {
            'test_loss': avg_test_loss,
            'avg_dice_score': avg_dice_score,
            'avg_hd_score': avg_hd_score,
            'avg_iou': avg_iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
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
