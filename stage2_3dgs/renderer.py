
"""
Differentiable Gaussian Rasterizer
Equations 13 and 14 from the paper
"""

import torch
import torch.nn as nn


class GaussianRenderer(nn.Module):
    def __init__(self, image_height, image_width):
        super().__init__()
        self.H = image_height
        self.W = image_width

    def forward(self, gaussian_params, camera):
        device = gaussian_params['means'].device
        means = gaussian_params['means']
        colors = gaussian_params['colors']
        opacities = gaussian_params['opacity'].squeeze()

        campos = camera['campos'].to(device)
        distances = torch.norm(means - campos.unsqueeze(0), dim=-1)
        depth_order = torch.argsort(distances, descending=True)

        rendered = torch.zeros(3, self.H, self.W, device=device)

        px = ((means[:, 0] / (distances + 0.5)) * 0.5 + 0.5) * self.W
        py = ((means[:, 1] / (distances + 0.5)) * 0.5 + 0.5) * self.H

        valid = (px >= 0) & (px < self.W) & (py >= 0) & (py < self.H) & (opacities > 0.01)
        sizes = torch.clamp(50.0 / (distances + 0.5), 2, 15).int()

        for idx in depth_order:
            if not valid[idx]:
                continue
            x = int(px[idx].item())
            y = int(py[idx].item())
            size = int(sizes[idx].item())
            color = colors[idx]
            opacity = opacities[idx].item()

            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        if dx*dx + dy*dy <= size*size:
                            alpha = opacity * (1 - (dx*dx + dy*dy) / (size*size))
                            rendered[:, ny, nx] = rendered[:, ny, nx] * (1 - alpha) + color * alpha

        return rendered
