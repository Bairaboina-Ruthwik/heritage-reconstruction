# stage1_sr/models/maet_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1_sr.models.macm import MACM
from stage1_sr.models.etm import ETM
from stage1_sr.models.hspam import HSPAM


class MAETNet(nn.Module):
    """
    Full MAET Super-Resolution Network.
    Figure 3 and Equations 7, 8, 9 in the paper.
    
    PIPELINE:
    1. Shallow feature extraction (3×3 conv) → F0
    2. MACM: multi-scale local texture extraction → FMACM
    3. ETM: global dependency modeling → FETM  
    4. HSPAM: sparse similarity-based attention → FHSPAM
    5. Fusion: Ffuse = FETM + α × FHSPAM  (Eq. 8)
    6. Reconstruction: ISR = Hup(Ffuse + F0) + f_up(ILR)  (Eq. 9)
       - Sub-pixel convolution for upsampling
       - Bilinear interpolation of LR as a residual
    
    The final upsampled image is the sum of:
    - The network's learned high-frequency detail
    - The bicubic-upsampled low-res image (baseline structure)
    This decomposition makes training easier since the network 
    only needs to learn the HIGH-FREQUENCY RESIDUAL, not the 
    whole image from scratch.
    """
    def __init__(self, num_feat=64, scale=4,
                 num_group=4, num_block=4):
        super().__init__()
        self.scale = scale
        
        # Step 1: Shallow feature extraction
        self.conv_first = nn.Conv2d(3, num_feat, 3, padding=1)
        
        # Step 2: MACM
        self.macm = MACM(num_feat, num_group, num_block)
        
        # Step 3: ETM
        self.etm = ETM(num_feat, num_feat)
        
        # Step 4: HSPAM
        self.hspam = HSPAM(num_feat)
        
        # Fusion weight α (Equation 8) — learnable
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # Step 6: High-resolution reconstruction
        # Adjust channels to 3 × scale² for sub-pixel conv
        self.conv_before_upsample = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.upsample_conv = nn.Conv2d(num_feat, 3 * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)  # sub-pixel convolution
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_lr):
        """
        x_lr: low-resolution input image [B, 3, H, W]
        returns: high-resolution output [B, 3, H*scale, W*scale]
        """
        # Step 1: Shallow features
        f0 = self.relu(self.conv_first(x_lr))   # [B, num_feat, H, W]
        
        # Step 2: MACM
        f_macm = self.macm(f0)                   # [B, num_feat, H, W]
        
        # Step 3: ETM (Equation 4)
        f_etm = self.etm(f_macm)                 # [B, num_feat, H, W]
        
        # Step 4: HSPAM (Equation 7)
        f_hspam = self.hspam(f_macm)             # [B, num_feat, H, W]
        
        # Step 5: Fuse (Equation 8)
        f_fuse = f_etm + self.alpha * f_hspam    # [B, num_feat, H, W]
        
        # Add long skip residual from shallow features (bottom of Fig. 3)
        f_fuse = f_fuse + f0
        
        # Step 6: Reconstruction (Equation 9)
        f_up = self.relu(self.conv_before_upsample(f_fuse))
        f_up = self.upsample_conv(f_up)
        i_sr_residual = self.pixel_shuffle(f_up)   # [B, 3, H*scale, W*scale]
        
        # Bilinear baseline: structure from bicubic upsampled LR
        i_lr_up = F.interpolate(x_lr, scale_factor=self.scale,
                                mode='bilinear', align_corners=False)
        
        # ISR = learned residual + bilinear baseline
        i_sr = i_sr_residual + i_lr_up
        return i_sr.clamp(0, 1)  # pixel values must be in [0, 1]