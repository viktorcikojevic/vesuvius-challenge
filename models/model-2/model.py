import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=1, init_features=64, class_one_weight=1):
        super(UNet, self).__init__()
        
        self.class_one_weight = class_one_weight
        
        # Load the pre-trained UNet model
        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=False)
        
    def f0point5_score(self, output, target):
        # Flatten the output and target tensors
        output = output.view(-1)
        target = target.view(-1)
        
        # Convert the output to binary values 0 and 1
        output = (output > 0.5).float()
        
        # Calculate the precision and recall
        tp = torch.sum(output * target)
        fp = torch.sum(output * (1 - target))
        fn = torch.sum((1 - output) * target)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        # Calculate the F0.5 score
        beta = 0.5
        f0point5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
        return f0point5
    
    def forward(self, x, targets=None):
        x = self.unet(x)
        # Convert the predictions to binary values 0 and 1
        predictions = (x > 0.5).float()
        
        out = {}
        
        if targets is None:
            loss = None
            precision = None
            accuracy = None
            f0point5 = None
        else:
            # Calculate the loss
            
            # Calculate class weights based on the imbalance
            class_weights = torch.tensor([1.0, self.class_one_weight]) # weight 1 for class 0, weight 5 for class 1

            # Instantiate the loss function
            loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights[1], reduction='mean')
            
            # Flatten the targets and x tensor
            targets = targets.view(-1)
            x = x.view(-1)
            predictions_flat = predictions.view(-1)
            
            
            # Calculate the loss
            loss = loss_function(x, targets)
            
            # Calculate the accuracy: number of correctly predicted pixel / total number of pixels
            # Calculate the accuracy
            accuracy = (predictions_flat == targets).float().mean()
            
            # Calculate the precision
            tp = torch.sum(predictions_flat * targets)   
            fp = torch.sum(predictions_flat * (1 - targets))
            fn = torch.sum((1 - predictions_flat) * targets)
            precision = tp / (tp + fp + 1e-7)
            
            # Calculate F0.5 score
            f0point5 = self.f0point5_score(predictions_flat, targets) 
        
        out = {
            "loss": loss,
            "logits": x,
            "predictions": predictions,
            "accuracy": accuracy,
            "precision": precision,
            "f0point5": f0point5
        }
        
        return out
