from kaggle_vesuvius.models import resnet_3d#, pvtv2, inceptionv4

def build_model(config):
    # Define the model
    assert config['model']['name'] in ['resnet_3d', 'pvtv2', 'inceptionv4'], f"Model {config['model']['name']} not implemented"
    
    z_start = config['z_start']
    z_end = config['z_end']
    
    
    if config['model']['name'] == 'resnet_3d':
        model = resnet_3d.Resnet3DSegModel(resnet_depth=config['model']['depth'])
    
    if config['model']['name'] == 'pvtv2':
        img_size = config['crop_size']
        in_chans = z_end - z_start
        num_classes = 1
        model = pvtv2.PyramidVisionTransformerV2(img_size=img_size, 
                                                 in_chans=in_chans, 
                                                 num_classes=num_classes)
        
    if config['model']['name'] == 'inceptionv4':
        in_channels = z_end - z_start
        model = inceptionv4.InceptionV4FPN3DResNet(in_channels=in_channels)
        
    return model