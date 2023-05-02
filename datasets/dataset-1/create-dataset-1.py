# create-dataset-1.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import albumentations as A
import argparse
import gc


def random_crop(volume, mask, label):
    # Note: image_data_format is 'channel_last'
    # take random number between 512 and 1024
    crop_size = np.random.randint(512, 1024)
    height, width = volume.shape[:-1]
    dx = crop_size
    dy = crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    volume_crop = volume[y:(y+dy), x:(x+dx), :]
    mask_crop = mask[y:(y+dy), x:(x+dx)]
    label_crop = label[y:(y+dy), x:(x+dx)]
    return volume_crop, mask_crop, label_crop


def apply_augmentation(augmentation, image, mask, label):
    augmented = augmentation(image=image, mask=mask, label=label)
    return augmented['image'], augmented['mask'], augmented['label']


def resize_to_768(volume, mask, label):
    resize = A.Compose([
        A.Resize(height=768, width=768, p=1)
    ], additional_targets={'label': 'mask'})
    # Apply the resize
    augmented = resize(image=volume, mask=mask, label=label)
    aug_volume, aug_mask, aug_label = augmented['image'], augmented['mask'], augmented['label']
    return aug_volume, aug_mask, aug_label

def augment(augmentation, volume, mask, label, train=True):
    # random crop
    v, m, l = random_crop(volume, mask, label)
    m_pixels_active = np.sum(m)
    m_pixels_total = m.shape[0] * m.shape[1]
    if m_pixels_active / m_pixels_total < 0.3:
        # if less than 30% of the pixels are active, then the image is mostly background
        # so we discard it and try again
        del v, m, l
        gc.collect()
        return augment(augmentation, volume, mask, label, train=train)
    
    # apply augmentation    
    if train:
        v, m, l = apply_augmentation(augmentation, v, m, l)
    # resize to 768x768
    v, m, l = resize_to_768(v, m, l)
    # normalize v
    v = (v - np.min(v)) / (np.max(v) - np.min(v))
    return v, m, l

def load_volume(surface_volume_path):
    img = []
    for i in tqdm(range(64)):
        x = np.array(Image.open(os.path.join(surface_volume_path, f'{i:02d}.tif')))
        img.append(x)
    img = np.array(img)
     # permute: (z, x, y) -> (x, y, z)
    img = np.transpose(img, (1, 2, 0))
    return img

def generate_augmented_dataset(seed=42, n_augmentations_per_scroll_train=10_000, n_augmentations_per_scroll_test=1000):
    # Set the random seed for numpy and Albumentations
    np.random.seed(seed)
    
    # Define the augmentation pipeline
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4, p=0.5)
    ], additional_targets={'label': 'mask'})
    
    
    rand_ints = [1, 2, 3]
    # shuffle the list
    np.random.shuffle(rand_ints)
    indx_train = 0
    indx_test = 0
    
    # if the folders train/volume, train/label, test/volume, test/label don't exist, create them. Otherwise delete them and create them again.
    if not os.path.exists('train/volume'):
        os.makedirs('train/volume')        
    if not os.path.exists('train/label'):
        os.makedirs('train/label')
    if not os.path.exists('test/volume'):
        os.makedirs('test/volume')
    if not os.path.exists('test/label'):
        os.makedirs('test/label')
    if not os.path.exists('train/volume-png'):
        os.makedirs('train/volume-png')
    if not os.path.exists('test/volume-png'):
        os.makedirs('test/volume-png')

        
    # loop over the train images.
    for rand_int in rand_ints:
        print("[INFO] Loading image", rand_int)
        surface_volume_path = f'../../kaggle-data/vesuvius-challenge-ink-detection/train/{rand_int}/surface_volume'
        volume = load_volume(surface_volume_path)
        mask = np.array(Image.open(f'../../kaggle-data/vesuvius-challenge-ink-detection/train/{rand_int}/mask.png'))
        ink_labels = np.array(Image.open(f'../../kaggle-data/vesuvius-challenge-ink-detection/train/{rand_int}/inklabels.png'))
        
        # train-test split
        print("[INFO] Splitting into train and test sets...")
        n_rows = volume.shape[0]
        n_train = int(n_rows * 0.75)
        n_test = int(n_rows * 0.8)
        
        volume_train = volume[:n_train]
        volume_test = volume[n_test:]
        mask_train = mask[:n_train]
        mask_test = mask[n_test:]
        ink_labels_train = ink_labels[:n_train]
        ink_labels_test = ink_labels[n_test:]
        
        # delete the original volume, mask and ink_labels to free up memory
        del volume, mask, ink_labels
        gc.collect()

        print("[INFO] Augmenting training data...")
        for _ in tqdm(range(n_augmentations_per_scroll_train)):
            v, m, l = augment(augmentation, volume_train, mask_train, ink_labels_train, train=True)
            # save the augmented volume
            fname_volume = f"train/volume/{indx_train}"
            fname_label = f"train/label/{indx_train}"
            
            # save the augmented volume and label as numpy arrays   
            np.save(fname_volume, v)
            np.save(fname_label, l)
            
            # save the channel 0 of the augmented volume as a png image to train/volume-png
            v = v[:, :, 0]
            v = (v * 255).astype(np.uint8)
            v = Image.fromarray(v)
            v.save(f"train/volume-png/{indx_train}.png")
            
            # save v and l
            indx_train += 1
        
        print("[INFO] 'Augmenting' (cropping) test data...")
        for _ in tqdm(range(n_augmentations_per_scroll_test)):
            v, m, l = augment(augmentation, volume_test, mask_test, ink_labels_test, train=False)
            # save the augmented volume
            fname_volume = f"test/volume/{indx_test}"
            fname_label = f"test/label/{indx_test}"
            
            # save the augmented volume and label as numpy arrays   
            np.save(fname_volume, v)
            np.save(fname_label, l)
            
            # save the channel 0 of the augmented volume as a png image to train/volume-png
            v = v[:, :, 0]
            v = (v * 255).astype(np.uint8)
            v = Image.fromarray(v)
            v.save(f"test/volume-png/{indx_test}.png")
            
            # save v and l
            indx_test += 1

        
    


def main():
    parser = argparse.ArgumentParser(description="Generate augmented dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument("--train-augm", type=int, default=10_000, help="Number of augmentations per scroll for training")
    parser.add_argument("--test-augm", type=int, default=1_000, help="Number of augmentations per scroll for testing")
    args = parser.parse_args()

    # Call the generate_augmented_dataset function with the provided seed value
    generate_augmented_dataset(seed=args.seed, n_augmentations_per_scroll_train=args.train_augm, n_augmentations_per_scroll_test=args.test_augm)


if __name__ == "__main__":
    main()
