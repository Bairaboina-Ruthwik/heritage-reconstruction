import torch
import torch.nn as nn
import numpy as np


class VoxelDecoder(nn.Module):
    def __init__(self, embed_dim=32, num_gaussians_per_voxel=8,
                 hidden_dim=128, view_dim=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_g = num_gaussians_per_voxel
        out_dim = num_gaussians_per_voxel * (1 + 3 + 6 + 3)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, voxel_embeddings, voxel_centers, voxel_scales, view_dirs):
        decoder_input = torch.cat([voxel_embeddings, view_dirs], dim=-1)
        raw = self.decoder(decoder_input)
        N = voxel_embeddings.shape[0]
        raw = raw.view(N, self.num_g, -1)
        opacity_raw = raw[..., :1]
        color_raw   = raw[..., 1:4]
        cov_raw     = raw[..., 4:10]
        offsets     = raw[..., 10:13]
        voxel_centers_exp = voxel_centers.unsqueeze(1)
        voxel_scales_exp  = voxel_scales.unsqueeze(1)
        means = voxel_centers_exp + voxel_scales_exp * offsets
        opacity = torch.sigmoid(opacity_raw)
        color   = torch.sigmoid(color_raw)
        cov_diag   = torch.exp(cov_raw[..., :3])
        cov_offdiag = cov_raw[..., 3:]
        return {
            'means':    means.view(-1, 3),
            'opacity':  opacity.view(-1, 1),
            'colors':   color.view(-1, 3),
            'cov_diag':    cov_diag.view(-1, 3),
            'cov_offdiag': cov_offdiag.view(-1, 3),
        }


def initialize_voxels_from_pointcloud(points, voxel_size=0.05, embed_dim=32):
    points_tensor = torch.tensor(points, dtype=torch.float32)
    voxel_indices = torch.floor(points_tensor / voxel_size).long()
    unique_voxels = torch.unique(voxel_indices, dim=0)
    N_voxels = len(unique_voxels)
    voxel_centers = (unique_voxels.float() + 0.5) * voxel_size
    voxel_embeddings = torch.randn(N_voxels, embed_dim) * 0.01
    voxel_scales = torch.ones(N_voxels, 1) * voxel_size
    print(f"Initialized {N_voxels:,} voxels from {len(points):,} points")
    return {
        'embeddings': nn.Parameter(voxel_embeddings),
        'centers':    voxel_centers,
        'scales':     voxel_scales,
    }

print("✅ voxel_decoder.py created")
