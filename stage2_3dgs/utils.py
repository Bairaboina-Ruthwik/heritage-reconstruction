
"""
Utility functions for camera loading and processing
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def create_cameras(num_cameras=20, image_size=(128, 128)):
    cameras = []
    to_tensor = transforms.ToTensor()
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 2.0
        position = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.5])
        img = Image.new('RGB', image_size, color=(128, 128, 128))
        img_tensor = to_tensor(img)
        cameras.append({
            'image': img_tensor,
            'position': position,
            'campos': torch.tensor(position, dtype=torch.float32),
        })
    return cameras
