import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Mapping between YAML names and actual loss classes
LOSS_MAP = {
    'bce': smp.losses.SoftBCEWithLogitsLoss,
    'dice': smp.losses.DiceLoss,
    'focal': smp.losses.FocalLoss(mode='binary')
}

def get_loss_function(loss_configs):
    # Create a list to hold individual loss functions and their weights
    losses = []
    weights = []
    
    for config in loss_configs:
        loss_name = config["name"]
        params = config.get("params", {})
        
        weight = params.get("weight", 1.0)
        weight = torch.tensor(weight)
        
        # Create the loss function
        if loss_name == 'dice':
            loss_fn = LOSS_MAP[loss_name](mode='binary', from_logits=True)
        if loss_name == 'bce':
            loss_fn = LOSS_MAP[loss_name]()
        
        losses.append(loss_fn)
        weights.append(weight)
    
    # Create the actual loss function
    def criterion(y_pred, y_true):
        loss = 0.0
        for loss_fn, weight in zip(losses, weights):
            loss += weight * loss_fn(y_pred, y_true)
        return loss
    
    return criterion