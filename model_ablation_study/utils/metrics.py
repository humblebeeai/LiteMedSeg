import torch
import numpy as np
import time
from scipy.spatial.distance import directed_hausdorff

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2 * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model, input_size, device):
    """Calculate FLOPs using thop library"""
    try:
        from thop import profile
        THOP_AVAILABLE = True
    except ImportError:
        THOP_AVAILABLE = False
        print("Warning: thop not available. Install with 'pip install thop' to calculate FLOPs")

    if not THOP_AVAILABLE:
        return None, None
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Calculate FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Convert to GFLOPs
    gflops = flops / 1e9
    flops = int(flops)
    
    return flops, gflops

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

def calculate_dice_score(outputs, masks):
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

def measure_inference_time(model, test_loader, device, num_warmup=5, num_measure=50):
    """Measure average inference time"""
    model.eval()
    device = torch.device(device)
    
    # Get a single batch for timing
    sample_batch = next(iter(test_loader))
    images, _ = sample_batch
    images = images.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(images)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_measure):
            start_time = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time
