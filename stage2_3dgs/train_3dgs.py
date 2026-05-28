
"""
Progressive 3-Stage 3DGS Training
Equations 15, 16, 17 from the paper
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveGaussianTrainer:
    def __init__(self, voxel_embeddings, voxel_centers, voxel_scales, 
                 decoder, renderer, cameras, device='cpu'):
        
        self.voxel_embeddings = nn.Parameter(voxel_embeddings.to(device))
        self.voxel_centers = voxel_centers.to(device)
        self.voxel_scales = voxel_scales.to(device)
        self.decoder = decoder.to(device)
        self.renderer = renderer
        self.cameras = cameras
        self.device = device

        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) + [self.voxel_embeddings],
            lr=0.0025, betas=(0.9, 0.99)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=1e-6)

    def get_gaussian_params(self, camera_idx):
        cam = self.cameras[camera_idx]
        cam_pos = torch.tensor(cam['position'], device=self.device, dtype=torch.float32)
        view_dirs = F.normalize(self.voxel_centers - cam_pos.unsqueeze(0), dim=-1)
        return self.decoder(self.voxel_embeddings, self.voxel_centers, self.voxel_scales, view_dirs)

    def stage1_loss(self, rendered, gt):
        return F.l1_loss(rendered, gt)

    def train(self, total_iters=2000, ckpt_dir='./checkpoints'):
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Starting progressive training for {total_iters} iterations")
        
        for iteration in range(total_iters):
            cam_idx = torch.randint(len(self.cameras), (1,)).item()
            cam = self.cameras[cam_idx]
            gauss_params = self.get_gaussian_params(cam_idx)
            rendered_img = self.renderer(gauss_params, cam)
            loss = self.stage1_loss(rendered_img, cam['image'].to(self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if iteration % 500 == 0 and iteration > 0:
                torch.save({
                    'iteration': iteration,
                    'loss': loss.item(),
                    'decoder_state': self.decoder.state_dict(),
                    'embeddings': self.voxel_embeddings.data,
                }, os.path.join(ckpt_dir, f'checkpoint_iter_{iteration}.pth'))
                print(f"💾 Checkpoint saved at iter {iteration}")
        
        print("✅ Training complete!")
