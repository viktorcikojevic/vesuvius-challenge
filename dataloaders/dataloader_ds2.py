from torch.nn.functional import avg_pool2d
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from torchvision import transforms
class ImageSegmentationDataset(Dataset):
    def __init__(self, root, mode='train', device='cpu', cache_refresh_interval=None, cache_n_images=64):
        
        assert mode in ['train', 'test'], "mode must be either 'train' or 'test'"
        
        self.root = root
        self.mode = mode
        self.cache_refresh_interval = cache_refresh_interval
        self.cache_n_images = cache_n_images
        self._load_data()
        
    def __len__(self):
        return len(self.volume_images)
    
    def _load_data(self):
        # Get the volume paths
        self.volume_images = os.listdir(os.path.join(self.root, self.mode, 'volume'))
        self.volume_images = [os.path.join(self.root, self.mode, 'volume', image) for image in self.volume_images]
        
        # Get label paths
        self.label_images = os.listdir(os.path.join(self.root, self.mode, 'label'))
        self.label_images = [os.path.join(self.root, self.mode, 'label', image) for image in self.label_images]
        
        # take self.cache_n_images random images from the dataset. They cannot be repeated
        if self.cache_n_images is not None and self.cache_n_images < len(self.volume_images):
            self.volume_images = np.random.choice(self.volume_images, size=self.cache_n_images, replace=False)
            self.label_images = np.random.choice(self.label_images, size=self.cache_n_images, replace=False)
        
        # Shuffle volume and label images simultaneouse
        self.volume_images, self.label_images = zip(*np.random.permutation(list(zip(self.volume_images, self.label_images))))
        
        # Load the data into memory
        self.cached_data = []
        for volume_image, label_image in zip(self.volume_images, self.label_images):
            try:
                volume = np.load(volume_image)
                label = np.load(label_image)
                
                if volume.shape != (768,768,16) or label.shape != (768,768):
                    print(f"Skipping {volume_image} and {label_image} because of wrong shape: {volume.shape} and {label.shape}")
                    continue
            except:
                print(f"Skipping {volume_image} and {label_image} because of error. Deleting those files")
                os.remove(volume_image)
                os.remove(label_image)                
                continue
            
            
            m, s = np.mean(volume, axis=(0, 1)), np.std(volume, axis=(0, 1))
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ])
            volume = preprocess(volume)
            volume = volume.permute(1, 2, 0)

            # convert to torch tensors
            label = torch.tensor(label, dtype=torch.float32)
            
            self.cached_data.append({
                "image": volume,
                "targets": label
            }) 
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, item):
        return self.cached_data[item]
    
    def reset(self):
        self._load_data()