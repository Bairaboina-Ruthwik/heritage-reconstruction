# stage1_sr/inference_sr.py

import sys
import os

# Point Python to the main 'heritage-reconstruction' folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from stage1_sr.models.maet_net import MAETNet

def evaluate_model(model, test_dir, scale=4, device='cuda'):
    """
    Evaluate PSNR and SSIM on a test directory.
    Computes on Y channel of YCbCr space (standard for SR benchmarks).
    """
    model.eval()
    to_tensor = transforms.ToTensor()
    
    psnr_list, ssim_list = [], []
    
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.png', '.jpg', 'bmp')):
            continue
        
        hr_img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
        
        # Crop to multiple of scale
        w, h = hr_img.size
        hr_img = hr_img.crop((0, 0, w - w % scale, h - h % scale))
        
        # Create LR version
        lr_img = hr_img.resize((hr_img.width // scale,
                                hr_img.height // scale), Image.BICUBIC)
        
        lr_t = to_tensor(lr_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr_t = model(lr_t)
        
        sr_img = transforms.ToPILImage()(sr_t.squeeze(0).cpu().clamp(0, 1))
        
        # Convert both to YCbCr and extract Y channel for metrics
        hr_y = np.array(hr_img.convert('YCbCr'))[:, :, 0].astype(np.float32)
        sr_y = np.array(sr_img.convert('YCbCr'))[:, :, 0].astype(np.float32)
        
        p = psnr_fn(hr_y, sr_y, data_range=255)
        s = ssim_fn(hr_y, sr_y, data_range=255)
        
        psnr_list.append(p)
        ssim_list.append(s)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"Results on {os.path.basename(test_dir)}: "
          f"PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# # Load model and evaluate
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MAETNet(num_feat=64, scale=4).to(device)

# ckpt = torch.load(r'D:\sr_iter_5000.pth',
#                   map_location=device)
# model.load_state_dict(ckpt['model'])

# # Paper target for ×4: Urban100 = 26.41 dB, Manga109 = 30.81 dB
# for dataset in ['Set5', 'Set14', 'B100', 'Urban100']:
#     evaluate_model(model, rf'C:\Users\RUTHWIK\OneDrive\Desktop\sr_benchmarks', scale=4)


# Load model and evaluate  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MAETNet(num_feat=64, scale=4).to(device)


ckpt = torch.load('/content/drive/MyDrive/heritage_reconstruction_copy/checkpoints/sr_model/sr_latest.pth', map_location=device)
model.load_state_dict(ckpt['model'])

for dataset in ['Set5', 'Set14', 'manga109', 'Urban100']:
    dataset_path = f'/content/drive/MyDrive/heritage_reconstruction_copy/sr_benchmarks/{dataset}/HR'
    
    # Optional safety check to prevent crashing if a folder is missing
    if os.path.exists(dataset_path):
        #evaluate_model(model, dataset_path, scale=4, device=device)
        # We catch the PSNR and SSIM returns here
        avg_psnr, avg_ssim = evaluate_model(model, dataset_path, scale=4, device=device)
        
        #  FIX: Print the actual dataset name here instead of inside the function
        print(f" Final Results on {dataset}: PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f}")
    else:
        print(f"Directory not found: {dataset_path}")

        