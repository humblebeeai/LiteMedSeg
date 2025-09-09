import os
from .utils.results import save_results_to_csv, plot_ablation_results, create_results_table
from .engine.trainer import AblationTrainer


def run_complete_ablation_study(train_loader, val_loader, test_loader, 
                               epochs=50, device='mps', save_path="./ablation_study"):
    """Run complete ablation study with training and evaluation"""
    
    trainer = AblationTrainer(device=device)

    os.makedirs(save_path, exist_ok=True)

    # Train all models
    print("Starting Complete Ablation Study Training...")
    training_results = trainer.train_all_ablation_models(
        train_loader, val_loader, epochs=epochs, save_path=f"{save_path}/models"
    )
    
    # Test all trained models
    print("\nEvaluating all models on test set...")
    test_results = []
    
    for model_name, result in training_results.items():
        model = result['model']
        # test_loss, test_dice, test_iou = trainer.validate_model(model, test_loader)

        model_results = trainer.validate_model(model, test_loader)

        test_result = {
            'model_name': model_name,
            'description': result['description'],
            'test_loss': model_results['test_loss'],
            'test_dice': model_results['avg_dice_score'],
            'avg_hd_score': model_results['avg_hd_score'],
            'test_iou': model_results['avg_iou'],
            'precision': model_results['precision'],
            'recall': model_results['recall'],
            'f1_score': model_results['f1_score'],
            'iou_batch_avg': model_results['iou_batch_avg'],
            'dice_batch_avg': model_results['dice_batch_avg'],
            'total_params': model_results['total_params'],
            'trainable_params': model_results['trainable_params'],
            # 'parameters': result['history']['parameters'],
            'avg_inference_time': model_results['avg_inference_time'],
            'fps': model_results['fps'],
            'flops': model_results['flops'],
            'gflops': model_results['gflops'],
            'val_dice': result['history']['best_dice'],
            'config': result['config']
        }
        save_results_to_csv(test_result, save_dir=f"{save_path}/by_model_evaluation_results", filename=model_name, timestamp=None)

        test_results.append(test_result)

        print(f"{model_name}: Test Dice={model_results['avg_dice_score']:.4f}, Test IoU={model_results['avg_iou']:.4f}, Params={test_result['trainable_params']:,}")

    # Plot results
    plot_ablation_results(training_results, save_path=f"{save_path}/plots")
    
    # Save detailed results
    import json
    with open(f"{save_path}/ablation_results.json", 'w') as f:
        # Convert results to JSON-serializable format
        json_results = []
        for result in test_results:
            json_result = result.copy()
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    # Create results table
    create_results_table(test_results, save_path=f"{save_path}/results_table.csv")
    
    return training_results, test_results
