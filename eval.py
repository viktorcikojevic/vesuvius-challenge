import torch
import numpy as np

@torch.no_grad()
def evaluate(model, dataloader, eval_steps, device):
    # Makes average of all the metrics across eval_steps batches
    
    # Set model to eval mode
    model.eval()

    # Initialize the metrics dictionary
    metrics = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "f0point5": []
    }
    
    for eval_step in range(eval_steps):
        batch = next(iter(dataloader))
        images, labels = batch['image'], batch['targets']
        # send to device
        images = images.to(device)
        labels = labels.to(device)
        
        images = images.permute(0, 3, 1, 2)
        out = model(images, labels)
        
        # Update the metrics
        for key in metrics.keys():
            metrics[key].append(out[key].detach().cpu().numpy())
        
    # Flatten (np.concatenate) each list in the metrics dictionary
    for key in metrics.keys():
        metrics[key] = [value.item() for value in metrics[key]]

    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
    model.train()

    # Add "val_" to the beginning of each key
    avg_metrics = {f"val_{key}": value for key, value in avg_metrics.items()}

    return avg_metrics
    
    
    return metrics