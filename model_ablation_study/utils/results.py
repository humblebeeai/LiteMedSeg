import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def save_results_to_csv(results, save_dir="./ablation_evaluation_results", filename='Unet_model_evaluation_results', timestamp=None):
    """Save evaluation results to CSV file"""
    
    if timestamp is not None:
        filename = f"{filename}_{timestamp}.csv"
    else:
        filename = f"{filename}.csv"

    # Prepare CSV data
    csv_data = [
        ['Metric', 'Value'],
        ['Model Name', results['model_name']],
        ['Description', results['description']],
        ['Test Loss', f"{results['test_loss']:.4f}"],
        ['Average Dice Score', f"{results['test_dice']:.4f}"],
        ['Average IoU', f"{results['test_iou']:.4f}"],
        ['Precision', f"{results['precision']:.4f}"],
        ['Recall', f"{results['recall']:.4f}"],
        ['F1 Score', f"{results['f1_score']:.4f}"],
        ['Val Dice', f"{results['val_dice']:.4f}"],
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

    os.makedirs(save_dir, exist_ok=True)

    # Write to CSV
    with open(os.path.join(save_dir, filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print(f"\nResults saved to: {save_dir}/{filename}")
    return filename

def plot_ablation_results(results, save_path="./plots"):
    """Plot comprehensive ablation study results"""
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract data
    model_names = []
    dice_scores = []
    iou_scores = []
    param_counts = []
    descriptions = []
    
    for name, result in results.items():
        model_names.append(name)
        dice_scores.append(result['history']['best_dice'])
        iou_scores.append(max(result['history']['val_ious']))
        param_counts.append(result['history']['parameters'])
        descriptions.append(result['description'])
    
    # 1. Performance comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(model_names))
    
    ax1.bar(x_pos, dice_scores, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x_pos, iou_scores, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('IoU Score')
    ax2.set_title('IoU Score Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Parameter vs Performance scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(param_counts, dice_scores, s=100, alpha=0.7, color='green')
    
    for i, name in enumerate(model_names):
        plt.annotate(name, (param_counts[i], dice_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Parameters (count)')
    plt.ylabel('Dice Score')
    plt.title('Parameters vs Performance Trade-off')
    # plt.ylim(0.85, 0.95)  # make y-axis to start at 85%
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_path}/params_vs_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Training curves for key models
    key_models = ['baseline', 'full_lightweight']
    
    plt.figure(figsize=(15, 5))
    
    for i, key_model in enumerate(key_models):
        if key_model in results:
            plt.subplot(1, 3, i+1)
            history = results[key_model]['history']
            epochs = range(1, len(history['val_dices']) + 1)
            
            plt.plot(epochs, history['train_losses'], label='Train Loss', alpha=0.7)
            plt.plot(epochs, history['val_losses'], label='Val Loss', alpha=0.7)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'{key_model} - Training Curves')
            plt.legend()
            plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()



def create_results_table(test_results, save_path):
    """Create a comprehensive results table for the paper"""
    import pandas as pd
    
    # Prepare data for table
    table_data = []
    for result in test_results:
        config = result['config']
        
        # Create component indicators
        lsk_indicator = "✓" if config.get('use_sk', False) else "✗"
        eca_indicator = "✓" if config.get('use_eca', False) else "✗"
        mpp_indicator = "✓" if config.get('use_mpp', False) else "✗"
        mda_indicator = "✓" if config.get('use_mda', False) else "✗"
        
        # # Add type indicators for standard vs lightweight
        # if config.get('use_sk', False):
        #     lsk_type = " (S)" if config.get('sk_type') == 'standard' else " (L)"
        #     lsk_indicator += lsk_type
        
        # if config.get('use_eca', False):
        #     eca_type = " (S)" if config.get('ca_type') == 'standard' else " (L)"
        #     eca_indicator += eca_type
            
        # if config.get('use_mpp', False):
        #     mpp_type = " (S)" if config.get('pp_type') == 'standard' else " (L)"
        #     mpp_indicator += mpp_type
            
        # if config.get('use_mda', False):
        #     mda_type = " (S)" if config.get('da_type') == 'standard' else " (L)"
        #     mda_indicator += mda_type
        
        table_data.append({
            'Model': result['model_name'],
            'Description': result['description'],
            'LSK': lsk_indicator,
            'ECA': eca_indicator,
            'MPP': mpp_indicator,
            'MDA': mda_indicator,
            'Parameters (M)': f"{result['trainable_params']/1e6:.2f}",
            'Val_Dice': f"{result['val_dice']:.4f}",
            'Test_Dice': f"{result['dice_batch_avg']:.4f}",
            'Test_IoU': f"{result['iou_batch_avg']:.4f}",
            'Test_Loss': f"{result['test_loss']:.4f}",
            'Infer_Time (ms)': f"{result['avg_inference_time']:.2f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1_Score': f"{result['f1_score']:.4f}",
            'FPS': f"{result['fps']:.2f}",
            'Flops': f"{result['flops']}",
            'GFlops': f"{result['gflops']:.2f}"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    df.to_csv(save_path, index=False)
    
    # Print formatted table
    print("\n" + "="*120)
    print("ABLATION STUDY RESULTS TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    print("Legend: ✓ = Included, ✗ = Not included")
    
    return df

def calculate_component_contributions(test_results):
    """Calculate individual component contributions"""
    
    # Find baseline result
    baseline_result = None
    for result in test_results:
        if result['model_name'] == 'baseline':
            baseline_result = result
            break
    
    if not baseline_result:
        print("Baseline model not found!")
        return
    
    baseline_dice = baseline_result['test_dice']
    
    # Calculate individual contributions
    contributions = {}
    individual_models = ['baseline_lsk', 'baseline_eca', 'baseline_mpp', 'baseline_mda']
    
    for model_name in individual_models:
        for result in test_results:
            if result['model_name'] == model_name:
                improvement = result['test_dice'] - baseline_dice
                component = model_name.replace('baseline_', '').upper()
                contributions[component] = improvement
                break
    
    # Find full model result
    full_result = None
    for result in test_results:
        if result['model_name'] == 'full_lightweight':
            full_result = result
            break
    
    if full_result:
        total_improvement = full_result['test_dice'] - baseline_dice
        individual_sum = sum(contributions.values())
        synergy = total_improvement - individual_sum
        
        print(f"\nCOMPONENT CONTRIBUTION ANALYSIS:")
        print(f"{'='*50}")
        print(f"Baseline Dice Score: {baseline_dice:.4f}")
        print(f"Individual Contributions:")
        for component, improvement in contributions.items():
            percentage = (improvement / baseline_dice) * 100
            print(f"  {component}: +{improvement:.4f} ({percentage:+.2f}%)")
        
        print(f"\nCombined Performance:")
        print(f"  Individual sum: +{individual_sum:.4f}")
        print(f"  Actual combined: +{total_improvement:.4f}")
        print(f"  Synergy effect: +{synergy:.4f}")
        
        synergy_percentage = (synergy / baseline_dice) * 100
        print(f"  Synergy contribution: {synergy_percentage:+.2f}%")
    
    return contributions

def analyze_efficiency_tradeoffs(test_results):
    """Analyze parameter efficiency vs performance tradeoffs"""
    
    # Calculate efficiency metrics
    efficiency_data = []
    
    for result in test_results:
        params_millions = result['parameters'] / 1e6
        dice_score = result['test_dice']
        
        # Calculate efficiency: Dice per million parameters
        efficiency = dice_score / params_millions if params_millions > 0 else 0
        
        efficiency_data.append({
            'model': result['model_name'],
            'description': result['description'],
            'params_m': params_millions,
            'dice': dice_score,
            'efficiency': efficiency
        })
    
    # Sort by efficiency
    efficiency_data.sort(key=lambda x: x['efficiency'], reverse=True)
    
    print(f"\nEFFICIENCY ANALYSIS (Dice Score per Million Parameters):")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Params(M)':<10} {'Dice':<8} {'Efficiency':<12} {'Description'}")
    print(f"{'-'*80}")
    
    for data in efficiency_data:
        print(f"{data['model']:<20} {data['params_m']:<10.2f} {data['dice']:<8.4f} "
              f"{data['efficiency']:<12.4f} {data['description']}")
    
    return efficiency_data