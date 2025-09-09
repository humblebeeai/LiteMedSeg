from .model import LightweightDeepSegNet

def create_ablation_models():
    """Factory function to create all ablation study variants"""
    
    configs = {
        # 1. Baseline
        'baseline': {
            'use_sk': False, 'use_eca': False, 'use_mpp': False, 'use_mda': False,
            'description': 'Standard U-Net style baseline with standard components'
        },
        
        # 2. Individual additions
        'baseline_lsk': {
            'use_sk': True, 'use_eca': False, 'use_mpp': False, 'use_mda': False,
            'description': 'Baseline + Lightweight SK'
        },
        'baseline_eca': {
            'use_sk': False, 'use_eca': True, 'use_mpp': False, 'use_mda': False,
            'description': 'Baseline + ECA'
        },
        'baseline_mpp': {
            'use_sk': False, 'use_eca': False, 'use_mpp': True, 'use_mda': False,
            'description': 'Baseline + Mobile Pyramid Pooling'
        },
        'baseline_mda': {
            'use_sk': False, 'use_eca': False, 'use_mpp': False, 'use_mda': True,
            'description': 'Baseline + Micro Deformable Attention'
        },
        
        # 3. Individual removals from full model
        'full_wo_lsk': {
            'use_sk': False, 'use_eca': True, 'use_mpp': True, 'use_mda': True,
            'description': 'Full model without LSK'
        },
        'full_wo_eca': {
            'use_sk': True, 'use_eca': False, 'use_mpp': True, 'use_mda': True,
            'description': 'Full model without ECA'
        },
        'full_wo_mpp': {
            'use_sk': True, 'use_eca': True, 'use_mpp': False, 'use_mda': True,
            'description': 'Full model without MPP'
        },
        'full_wo_mda': {
            'use_sk': True, 'use_eca': True, 'use_mpp': True, 'use_mda': False,
            'description': 'Full model without MDA'
        },
        
        # 4. Component combinations
        'encoder_components': {
            'use_sk': True, 'use_eca': True, 'use_mpp': False, 'use_mda': False,
            'description': 'LSK + ECA only'
        },
        'bottleneck_components': {
            'use_sk': False, 'use_eca': False, 'use_mpp': True, 'use_mda': True,
            'description': 'MPP + MDA only'
        },
        
        # 5. Lightweight vs Standard comparison
        'full_lightweight': {
            'use_sk': True, 'use_eca': True, 'use_mpp': True, 'use_mda': True,
            'description': 'Full lightweight model (proposed)'
        },
        
    }

    models = {}
    for name, config in configs.items():
        desc = config.pop('description')
        models[name] = {
            'model': LightweightDeepSegNet(**config),
            'description': desc,
            'config': config.copy()
        }
        config['description'] = desc  # Add it back
        
    return models