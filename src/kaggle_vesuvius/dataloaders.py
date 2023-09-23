from typing import *
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset
import os
import zarr
import tifffile
import imageio
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

class VesuviusDataset(Dataset):
    
    
    def get_zarr_file(self, folder: str, z_start: int = 0, z_end: int = None) -> zarr.core.Array:
        """
        Create an in-memory Zarr store from a given folder.
        zarr .volume is uint16, mask and label are uint8
        """

        # Paths to TIFF, mask, and labels
        tif_folder = f"{folder}/surface_volume"
        mask_path = f"{folder}/mask.png"
        label_path = f"{folder}/inklabels.png" if os.path.exists(f"{folder}/inklabels.png") else None

        # Read mask and labels
        mask_data = imageio.imread(mask_path, pilmode='L')
        label_data = imageio.imread(label_path, pilmode='L') if label_path else None


        # Initialize in-memory Zarr store
        store = zarr.MemoryStore()
        root = zarr.group(store=store)

        # Create Zarr array for mask
        mask_zarr = root.create_dataset("mask", shape=mask_data.shape, chunks=mask_data.shape, dtype=mask_data.dtype)
        mask_zarr[:] = mask_data

        # Create Zarr array for label if it exists
        if label_data is not None:
            label_zarr = root.create_dataset("label", shape=label_data.shape, chunks=label_data.shape, dtype=label_data.dtype)
            label_zarr[:] = label_data

        # Get list of TIFF files and apply slicing
        tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])[z_start:z_end]

        # Read first TIFF to determine shape and dtype
        first_tif_path = os.path.join(tif_folder, tif_files[0])
        first_tif = tifffile.imread(first_tif_path)

        # Create 3D Zarr array for volume
        z_shape, y_shape, x_shape = len(tif_files), first_tif.shape[0], first_tif.shape[1]
        volume = root.create_dataset("volume", shape=(z_shape, y_shape, x_shape), chunks=(1, y_shape, x_shape), dtype=first_tif.dtype)

        # Read each TIFF file and write to Zarr array
        print(f"Reading TIFF files from folder {tif_folder}")
        for i, filename in tqdm(enumerate(tif_files), total=z_shape):
            tif_path = os.path.join(tif_folder, filename)
            tif_data = tifffile.imread(tif_path)

            # Make sure to validate the shape and dtype of each TIFF if needed
            assert tif_data.shape == (y_shape, x_shape)
            assert tif_data.dtype == first_tif.dtype

            # Write the TIFF data to the Zarr array
            volume[i, :, :] = tif_data

        return root


    
    def __init__(self, 
                 mode: str,
                    data_dir: str,
                    crop_size: int,
                    eval_on: int,
                    z_start: int = 0,
                    z_end: int = None,
                    stride: int = 256,
                    **kwargs):
        self.fragments = kwargs.get("fragments", None)
        self.mode = mode
        self.crop_size = crop_size
        
        input_folders = sorted(os.listdir(data_dir))
        print(input_folders)
        data_dir = [os.path.join(data_dir, folder) for folder in input_folders]   
        indices = [int(indx.split("/")[-1]) for indx in data_dir]
        
        if self.mode == 'train':
            # remove value eval_on from data_dir
            data_dir = [folder for folder in data_dir if int(folder.split("/")[-1]) != eval_on]
            indices = np.array(indices)[np.array(indices) != eval_on]
        else:
            # take only index eval_on from data_dir
            data_dir = [folder for folder in data_dir if int(folder.split("/")[-1]) == eval_on]
            indices = np.array(indices)[np.array(indices) == eval_on]
        
        print(indices)
        n_channels = z_end - z_start
        
        self.fragments_zarr = {
            str(indx): self.get_zarr_file(folder, z_start, z_end) for indx, folder in zip(indices, data_dir)
        }
        fragments_shape = {k : v.mask.shape for k, v in self.fragments_zarr.items()}
        print(fragments_shape) # for eval_on = 1, {'0': (14830, 9506), '2': (7606, 5249)}
    
        print(f"Fragments shape is {fragments_shape}")
        
        self.xys = []
        
        print(f"Creating {self.mode} dataset")
        for fragment in self.fragments_zarr.keys():
            H, W = fragments_shape[fragment]
            for y in tqdm(range(0, H-self.crop_size+1, stride)):
                for x in range(0, W-self.crop_size+1, stride):
                    if self.mode == 'train':
                        mask = self.fragments_zarr[fragment].mask[y:y+self.crop_size, x:x+self.crop_size]
                        # Ignore the crop if it contains less than 20% of the mask
                        if np.sum(mask) / np.prod(mask.shape) > 0.2:
                            self.xys.append((fragment, x, y, W, H))
                    else:
                        self.xys.append((fragment, x, y, W, H))
        print(f"Dataset created. In total, n_crops={len(self.xys)}")
        
        size = self.crop_size
        self.train_aug_list = [
                A.Resize(size, size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(max_holes=8, 
                                max_width=int(self.crop_size * 0.1), 
                                max_height=int(self.crop_size * 0.1), 
                                mask_fill_value=0, p=0.5),
                
                A.Rotate(limit=90, p=0.9), 
                
                # A.Cutout(max_h_size=int(self.crop_size * 0.2),
                #          max_w_size=int(self.crop_size * 0.2), num_holes=1, p=1.0),
                A.Normalize(
                    mean= [0] * n_channels,
                    std= [1] * n_channels
                ),
                ToTensorV2(transpose_mask=True),
            ]
        
        # Create the augmentation pipeline
        if self.mode == "train":
            self.augmentations = A.Compose(self.train_aug_list)
        else:
            self.augmentations = A.Compose([
                A.Resize(size, size, p=1.0),
                A.Normalize(
                    mean= [0] * n_channels,
                    std= [1] * n_channels
                ),
                ToTensorV2(transpose_mask=True),
            ])
        
       
    def cut_mix(self, frag_crop, mask_crop): 
        
        # implement cutmix. 
        if np.random.uniform(0, 1) < 0.5:
            # get a random int between 0 and len(self.xys)
            random_index = np.random.randint(0, len(self.xys))
            fragment, x1, y1, W, H = self.xys[random_index]
            
            x2 = x1 + self.crop_size
            y2 = y1 + self.crop_size
            

            frag_crop_2 = self.fragments_zarr[fragment].volume[:, y1:y2, x1:x2]            
            mask_crop_2 = self.fragments_zarr[fragment].label[y1:y2, x1:x2]
            frag_crop_2 = frag_crop/255.0 # "convert" to uint8
            
            
            
            rand_wh = np.random.randint(32, self.crop_size//2)
            rand_position_start_y = np.random.randint(0, self.crop_size-rand_wh)
            rand_position_start_x = np.random.randint(0, self.crop_size-rand_wh)
            rand_weight = np.random.uniform(0.2, 0.8)
            
            frag_crop[rand_position_start_y:rand_position_start_y+rand_wh, 
                      rand_position_start_x:rand_position_start_x+rand_wh, 
                      :] = frag_crop_2[rand_position_start_y:rand_position_start_y+rand_wh, 
                                       rand_position_start_x:rand_position_start_x+rand_wh, :] * rand_weight \
                        + frag_crop[rand_position_start_y:rand_position_start_y+rand_wh, 
                                    rand_position_start_x:rand_position_start_x+rand_wh, :] * (1-rand_weight)
            
            # same for the mask
            mask_crop[rand_position_start_y:rand_position_start_y+rand_wh,
                        rand_position_start_x:rand_position_start_x+rand_wh] = mask_crop_2[rand_position_start_y:rand_position_start_y+rand_wh,
                                                                                            rand_position_start_x:rand_position_start_x+rand_wh] * rand_weight \
                        + mask_crop[rand_position_start_y:rand_position_start_y+rand_wh,
                                    rand_position_start_x:rand_position_start_x+rand_wh] * (1-rand_weight)
                        
            
        return frag_crop, mask_crop
        
    def __getitem__(self, i):
        fragment, x1, y1, W, H = self.xys[i]
        
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        
        frag_crop = (self.fragments_zarr[fragment].volume[:, y1:y2, x1:x2] / 255.0).astype('float32') # "convert" to uint8
        mask_crop = (self.fragments_zarr[fragment].label[y1:y2, x1:x2] / 255.0).astype('int32')
        
        # swap axes: [C, H, W] -> [H, W, C]
        frag_crop = np.moveaxis(frag_crop, 0, -1)
        
        original_hw = frag_crop.shape[:2]
        
        if original_hw[0] == 0 or original_hw[1] == 0:
            raise ValueError(f"original_hw is {original_hw}")
        
        
        # Apply the augmentations
        augmented = self.augmentations(image=frag_crop, mask=mask_crop)
        
        # Separate the image and mask
        frag_crop, mask_crop = augmented["image"], augmented["mask"]
        
        # # Apply cutmix
        # if self.mode == 'train':
        #     frag_crop, mask_crop = self.cut_mix(frag_crop, mask_crop)

        frag_crop = frag_crop.unsqueeze(0)
        mask_crop = mask_crop.float()
        
        # Resize using F interpolation back to original_hw
        frag_crop = F.interpolate(frag_crop, size=original_hw, mode="bilinear", align_corners=False)
        # mask_crop: [H, W] -> [1, H, W] -> [1, H_og, W_og]
        mask_crop = F.interpolate(mask_crop.unsqueeze(0).unsqueeze(0), size=original_hw, mode="nearest").squeeze()

        frag_crop = frag_crop.squeeze(0)
        
        return frag_crop, mask_crop

    def __len__(self):
        return 10 # len(self.xys)