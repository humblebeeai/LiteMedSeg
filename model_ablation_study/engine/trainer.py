import torch
from ..utils.metrics import count_parameters
from ..utils.loss import LightweightLoss
from ..models.config import create_ablation_models
from .validator import Validator

class AblationTrainer(Validator):
    """Training class for ablation study models"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.criterion = LightweightLoss()
        
    def train_model(self, model, train_loader, val_loader, epochs=50, lr=1e-3, 
                   model_name="model", save_path=None):
        """Train a single ablation model"""
        
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_losses = []
        val_dices = []
        val_ious = []
        
        best_dice = 0.0
        best_model_state = None
        
        print(f"Training {model_name}...")
        print(f"Parameters: {count_parameters(model):,}")
        print("-" * 50)
        
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
                
                # Print progress
                if batch_idx % 20 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Validation phase
            val_results = self.validate_model(model, val_loader)
            val_loss = val_results['test_loss']
            val_dice = val_results['avg_dice_score']
            val_iou = val_results['avg_iou']

            # Learning rate scheduling
            scheduler.step()
            
            # Record metrics
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_state = model.state_dict().copy()
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Dice: {val_dice:.4f}, '
                  f'Val IoU: {val_iou:.4f}')
            print("-" * 50)
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save model if path provided
        if save_path:
            torch.save({
                'model_state_dict': best_model_state,
                'model_name': model_name,
                'best_dice': best_dice,
                'config': model.config if hasattr(model, 'config') else None
            }, f"{save_path}/{model_name}_best.pth")
        
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'val_ious': val_ious,
            'best_dice': best_dice,
            'parameters': count_parameters(model)
        }
        
        return model, training_history

    def train_all_ablation_models(self, train_loader, val_loader, epochs=100, save_path="./ablation_models"):
        """Train all ablation study models"""
        models = create_ablation_models()
        results = {}
        
        import os
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for model_name, model_info in models.items():
            print(f"\n{'='*60}")
            print(f"Training: {model_info['description']}")
            print(f"{'='*60}")
            
            model = model_info['model']
            model.config = model_info['config']  # Add config for saving
            
            trained_model, history = self.train_model(
                model, train_loader, val_loader, epochs, 
                model_name=model_name, save_path=save_path
            )
            
            results[model_name] = {
                'model': trained_model,
                'history': history,
                'description': model_info['description'],
                'config': model_info['config']
            }
            
            print(f"Completed: {model_name}")
            print(f"Best Dice: {history['best_dice']:.4f}")
            print(f"Parameters: {history['parameters']:,}")
        
        return results
