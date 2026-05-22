# stage1_sr/train_sr.py

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse

from stage1_sr.models.maet_net import MAETNet
from stage1_sr.dataset import get_dataloader


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Model
    model = MAETNet(
        num_feat=64, scale=config['scale'],
        num_group=4, num_block=4
    ).to(device)
    
    # Optimizer (Adam with cosine schedule as described in paper)
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config['total_iters'],
                                  eta_min=1e-7)
    
    # Loss: L1 as stated in Equation 10
    criterion = nn.L1Loss()
    
    # DataLoader
    loader = get_dataloader(
        config['data_dir'],
        patch_size=48 * config['scale'],
        scale=config['scale'],
        batch_size=config['batch_size']
    )
    
    # Resume from checkpoint if one exists
    start_iter = 0
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    ckpt_path = os.path.join(config['ckpt_dir'], 'sr_latest.pth')
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iteration']
        print(f"✅ Resumed from iteration {start_iter}")
    else:
        print("Starting fresh training")
    
    # Training loop
    model.train()
    data_iter = iter(loader)
    
    for iteration in range(start_iter, config['total_iters']):
        try:
            lr_imgs, hr_imgs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            lr_imgs, hr_imgs = next(data_iter)
        
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # Forward pass
        sr_imgs = model(lr_imgs)
        
        # Compute L1 loss (Equation 10)
        loss = criterion(sr_imgs, hr_imgs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if iteration % 100 == 0:
            print(f"Iter {iteration}/{config['total_iters']} | "
                  f"Loss: {loss.item():.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # ─── CHECKPOINT every 500 iterations ─────────────────────
        # CRITICAL: This saves to Google Drive so either person can 
        # pick up training from where the other left off.
        if iteration % 500 == 0 and iteration > 0:
            torch.save({
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'loss':      loss.item(),
            }, ckpt_path)
            
            # Also save a milestone copy every 5000 iterations
            if iteration % 5000 == 0:
                milestone_path = os.path.join(
                    config['ckpt_dir'], f'sr_iter_{iteration}.pth'
                )
                torch.save({'model': model.state_dict(),
                            'iteration': iteration}, milestone_path)
                print(f"💾 Milestone checkpoint saved: {milestone_path}")
            else:
                print(f"💾 Checkpoint saved at iter {iteration}")
    
    print("Training complete!")


if __name__ == '__main__':
    # ─── COLAB USAGE ─────────────────────────────────────────────
    # Adjust paths to your Google Drive folder
    config = {
        'data_dir':    '/content/drive/MyDrive/heritage_reconstruction_shared/data/DIV2K/DIV2K_train_HR',
        'ckpt_dir':    '/content/drive/MyDrive/heritage_reconstruction_shared/checkpoints/sr_model',
        'total_iters': 250000,    # paper trains for 250k iterations
        'batch_size':  16,        # paper uses 16
        'scale':       4,         # ×4 super-resolution (as in paper results)
    }
    train(config)