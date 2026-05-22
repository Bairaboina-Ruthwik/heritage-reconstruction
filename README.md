# heritage-reconstruction

Alternate codes  
OG: !pip install -r requirements.txt --quiet
ALT: 
```txt
%%writefile requirements.txt
torch
torchvision
opencv-python
# Deep Learning Core
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Image Processing
Pillow==10.0.0
opencv-python==4.8.0.76
imageio==2.31.1
scikit-image==0.21.0

# Data & Math
numpy==1.24.3
scipy==1.11.1
einops==0.7.0

# 3DGS Dependencies
plyfile==0.9
open3d==0.17.0
roma==1.2.4

# Visualization & Logging
matplotlib==3.7.2
tqdm==4.65.0
tensorboard==2.14.0

# Structure from Motion
hloc @ git+https://github.com/cvg/Hierarchical-Localization.git

# Metrics
lpips==0.1.4
pytorch-msssim==0.2.1
```


