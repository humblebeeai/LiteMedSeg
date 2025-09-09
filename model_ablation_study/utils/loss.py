import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightLoss(nn.Module):
    """Simplified loss for lightweight training"""
    def __init__(self, main_weight=0.6, aux_weight=0.4):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.bce_dice = BCEDiceLoss()
    
    def forward(self, outputs, targets):
        # Ensure targets are float and have correct shape
        if targets.dtype != torch.float32:
            targets = targets.float()
        
        if isinstance(outputs, tuple):
            # Training mode with auxiliary outputs
            main_out, aux4, aux3 = outputs
            
            # Main loss
            main_loss = self.bce_dice(main_out, targets)
            
            # Auxiliary losses - resize targets if needed
            if aux4.shape != targets.shape:
                targets_aux4 = F.interpolate(targets, size=aux4.shape[-2:], mode='nearest')
            else:
                targets_aux4 = targets
                
            if aux3.shape != targets.shape:
                targets_aux3 = F.interpolate(targets, size=aux3.shape[-2:], mode='nearest')
            else:
                targets_aux3 = targets
            
            aux_loss4 = self.bce_dice(aux4, targets_aux4)
            aux_loss3 = self.bce_dice(aux3, targets_aux3)
            aux_loss = (aux_loss4 + aux_loss3) / 2
            
            total_loss = self.main_weight * main_loss + self.aux_weight * aux_loss
            return total_loss
        else:
            # Inference mode with single output
            return self.bce_dice(outputs, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)
    

def create_loss_function(loss_type='lightweight'):
    """
    Factory function to create appropriate loss function
    
    Args:
        loss_type: 'simple', 'lightweight', 'bce_dice'
    """
    if loss_type == 'simple':
        return BCEDiceLoss()
    elif loss_type == 'lightweight':
        return LightweightLoss()
    elif loss_type == 'bce_dice':
        return BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
