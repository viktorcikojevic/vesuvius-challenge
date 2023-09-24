import torch
import torch.nn as nn
from segmentation_models_pytorch import FPN

from .resnet_3d import Resnet3DDecoder

class InceptionV4FPN3DResNet(nn.Module):
    def __init__(self, in_channels, pooling):
        super(InceptionV4FPN3DResNet, self).__init__()

        # Encoder
        self.encoder = FPN(encoder_name="inceptionv4", encoder_weights=None, in_channels=in_channels, classes=1).encoder

        # Decoder
        encoder_dims = [8, 64, 192, 384, 1024, 1536]  # These are the output dimensions from your encoder
        self.decoder = Resnet3DDecoder(encoder_dims=encoder_dims, upscale=1)
        
        self.pooling = pooling
        assert self.pooling in ['mean', 'max']

    def forward(self, x):
        feat_maps = self.encoder(x) # returns a list of feature maps
        if self.pooling == 'mean':
            feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        else:
            feat_maps_pooled = [torch.max(f, dim=1) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
