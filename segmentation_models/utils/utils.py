import torch
import time
import os
import csv
from datetime import datetime

# Try to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with 'pip install thop' to calculate FLOPs")

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_size, device):
    """Calculate FLOPs using thop library"""
    if not THOP_AVAILABLE:
        return None, None
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Calculate FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Convert to GFLOPs
    gflops = flops / 1e9
    
    return flops, gflops

def measure_inference_time(model, test_loader, device, num_warmup=5, num_measure=50):
    """Measure average inference time"""
    model.eval()
    
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

def save_results_to_csv(results, folder_name="evaluation_results", filename='model_evaluation_results', timestamp=None):
    """Save evaluation results to CSV file"""
    
    # Add timestamp to filename
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}.csv"
    
    # Prepare CSV data
    csv_data = [
        ['Metric', 'Value'],
        ['Test Loss', f"{results['test_loss']:.4f}"],
        ['Average Dice Score', f"{results['avg_dice_score']:.4f}"],
        ['Average Hausdorff Distance', f"{results['avg_hd_score']:.4f}"],
        ['Average IoU', f"{results['avg_iou']:.4f}"],
        ['Precision', f"{results['precision']:.4f}"],
        ['Recall', f"{results['recall']:.4f}"],
        ['F1 Score', f"{results['f1']:.4f}"],
        ['IoU (batch avg)', f"{results['iou_batch_avg']:.4f}"],
        ['Dice (batch avg)', f"{results['dice_batch_avg']:.4f}"],
        ['Total Parameters', f"{results['total_params']:,}"],
        ['Trainable Parameters', f"{results['trainable_params']:,}"],
        ['Average Inference Time (seconds)', f"{results['avg_inference_time']:.6f}"],
        ['FPS (Frames Per Second)', f"{results['fps']:.2f}"],
    ]
    
    if results['flops'] is not None:
        csv_data.extend([
            ['FLOPs', f"{results['flops']:,}"],
            ['GFLOPs', f"{results['gflops']:.4f}"],
        ])

    # Print results
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")

    for row in csv_data[1:]:
        print(f"{row[0]:<35}: {row[1]}")

    print(f"{'='*50}")

    os.makedirs(folder_name, exist_ok=True)

    # Write to CSV
    with open(os.path.join(folder_name, filename_with_timestamp), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print(f"\nResults saved to: {folder_name}/{filename_with_timestamp}")
    return filename_with_timestamp