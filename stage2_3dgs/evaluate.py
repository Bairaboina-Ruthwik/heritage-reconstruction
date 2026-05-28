
"""
Day 8: Novel View Synthesis and Evaluation
Computes PSNR, SSIM, LPIPS metrics on actual images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import lpips
from pytorch_msssim import ssim as ssim_fn
import math
import os


class NovelViewEvaluator:
    def __init__(self, decoder, renderer, device='cpu'):
        self.decoder = decoder
        self.renderer = renderer
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    def compute_metrics(self, rendered, ground_truth):
        rendered = rendered.clamp(0, 1)
        gt = ground_truth.clamp(0, 1)
        
        mse = torch.mean((rendered - gt) ** 2)
        psnr = 10 * math.log10(1.0 / (mse.item() + 1e-8))
        ssim_val = ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0), data_range=1.0).item()
        
        rendered_lpips = rendered * 2 - 1
        gt_lpips = gt * 2 - 1
        lpips_val = self.lpips_fn(rendered_lpips.unsqueeze(0), gt_lpips.unsqueeze(0)).item()
        
        return psnr, ssim_val, lpips_val
    
    def render_novel_views(self, embeddings, centers, scales, cameras):
        rendered_images = []
        with torch.no_grad():
            for cam in cameras:
                cam_pos = cam['campos'].to(self.device)
                view_dirs = F.normalize(centers - cam_pos.unsqueeze(0), dim=-1)
                gauss_params = self.decoder(embeddings, centers, scales, view_dirs)
                rendered = self.renderer(gauss_params, cam)
                rendered_images.append(rendered)
        return rendered_images
    
    def evaluate(self, embeddings, centers, scales, cameras, ground_truths):
        psnr_list, ssim_list, lpips_list = [], [], []
        
        for i, (cam, gt) in enumerate(zip(cameras, ground_truths)):
            cam_pos = cam['campos'].to(self.device)
            view_dirs = F.normalize(centers - cam_pos.unsqueeze(0), dim=-1)
            gauss_params = self.decoder(embeddings, centers, scales, view_dirs)
            rendered = self.renderer(gauss_params, cam)
            
            psnr, ssim_val, lpips_val = self.compute_metrics(rendered, gt)
            psnr_list.append(psnr)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
        
        return {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'lpips': np.mean(lpips_list),
            'per_image': list(zip(psnr_list, ssim_list, lpips_list))
        }
