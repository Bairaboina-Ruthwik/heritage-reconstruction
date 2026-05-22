# stage1_sr/dataset.py

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DIV2KDataset(Dataset):
    """
    DIV2K Super-Resolution Dataset Loader.
    
    PURPOSE: The SR network needs pairs of (low-res, high-res) images.
    We only STORE the high-res images; the low-res versions are 
    generated ON THE FLY by bicubic downsampling.
    
    PATCH-BASED TRAINING: Instead of training on full images (too large for 
    GPU memory), we randomly crop 48×48 patches from the HR image, then 
    downsample to get the LR patch. This means each training step sees a 
    different crop — effectively infinite data augmentation.
    
    DATA AUGMENTATION: Random flips and rotations (90/180/270°) to 
    prevent the network from overfitting to one orientation.
    """
    def __init__(self, hr_dir, patch_size=48, scale=4, augment=True):
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        
        self.image_paths = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Dataset loaded: {len(self.image_paths)} images from {hr_dir}")

    def __len__(self):
        # Each image contributes multiple patches per epoch
        return len(self.image_paths) * 20

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        hr_img = Image.open(self.image_paths[img_idx]).convert('RGB')
        
        # Random crop a patch_size×patch_size region from HR
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            hr_img = hr_img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC
            )
            w, h = hr_img.size
        
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr_patch = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        # Create LR patch by downsampling
        lr_size = self.patch_size // self.scale
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)
        
        # Data augmentation
        if self.augment:
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                hr_patch = hr_patch.rotate(rot)
                lr_patch = lr_patch.rotate(rot)
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensors [C, H, W], values in [0, 1]
        to_tensor = transforms.ToTensor()
        return to_tensor(lr_patch), to_tensor(hr_patch)


def get_dataloader(hr_dir, patch_size=48, scale=4,
                   batch_size=16, num_workers=2):
    dataset = DIV2KDataset(hr_dir, patch_size, scale)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers,
                      pin_memory=True)