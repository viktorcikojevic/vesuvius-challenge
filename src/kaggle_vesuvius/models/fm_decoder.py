import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class FeatureMapsDecoder(nn.Module):
    def __init__(self, encoder_dims, upscale, pooling='mean'):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")
        self.pooling = pooling
        assert self.pooling in ['mean', 'max']

    def perform_pooling(self, feat_maps):
        
        # feat_maps is of shape (B, C, D, H, W): B is the batch size, C is the number of channels, D is the depth, H is the height, W is the width
        # dim=2 is the depth dimension
        
        if self.pooling == 'mean':
            feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        else:
            feat_maps_pooled = [torch.max(f, dim=2) for f in feat_maps]
            
        return feat_maps_pooled
        

    def forward(self, feature_maps):
        
        feature_maps = self.perform_pooling(feature_maps)
        
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1) #  Concatenate the upsampled feature map with the shallower feature map
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        
        mask = mask.squeeze()
        
        return mask