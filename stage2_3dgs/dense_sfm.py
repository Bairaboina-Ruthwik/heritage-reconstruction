# Run this cell FIRST to create dense_sfm.py

import os
import numpy as np
from pathlib import Path
import torch

class DenseSfMInitializer:
    """
    Dense-SfM Initialization Module.
    Takes SR-enhanced images → estimates camera poses + dense point cloud.
    """
    
    def __init__(self, image_dir, output_dir, use_gpu=True):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu

    def run(self):
        """Run full Dense-SfM pipeline."""
        print("Step 1: Extracting features with SuperPoint...")
        self._extract_features()
        
        print("Step 2: Matching features with SuperGlue...")
        self._match_features()
        
        print("Step 3: Running COLMAP SfM...")
        self._run_colmap()
        
        print("Step 4: Loading point cloud...")
        points = self._load_point_cloud()
        
        print(f"✅ Dense-SfM complete: {len(points):,} 3D points")
        return {
            'points': points,
            'sfm_dir': self.output_dir / 'sfm'
        }

    def _extract_features(self):
        from hloc import extract_features
        feature_conf = extract_features.confs['superpoint_aachen']
        self.feature_path = self.output_dir / 'features.h5'
        extract_features.main(
            feature_conf,
            self.image_dir,
            feature_path=self.feature_path
        )

    def _match_features(self):
        from hloc import match_features, pairs_from_exhaustive
        
        self.pairs_path = self.output_dir / 'pairs.txt'
        image_list = [str(p.name) for p in sorted(self.image_dir.glob('*.jpg')) + 
                      sorted(self.image_dir.glob('*.png'))]
        pairs_from_exhaustive.main(self.pairs_path, image_list=image_list)
        
        match_conf = match_features.confs['superglue']
        self.match_path = self.output_dir / 'matches.h5'
        match_features.main(
            match_conf,
            self.pairs_path,
            features=self.feature_path,
            matches=self.match_path
        )

    def _run_colmap(self):
        from hloc import reconstruction
        self.sfm_dir = self.output_dir / 'sfm'
        reconstruction.main(
            self.sfm_dir,
            self.image_dir,
            self.pairs_path,
            self.feature_path,
            self.match_path
        )

    def _load_point_cloud(self):
        """Load 3D points from COLMAP output."""
        import open3d as o3d
        ply_path = self.sfm_dir / 'points3D.ply'
        if ply_path.exists():
            pcd = o3d.io.read_point_cloud(str(ply_path))
            return np.asarray(pcd.points)
        else:
            return self._read_colmap_binary()

    def _read_colmap_binary(self):
        """Fallback: read COLMAP points3D.bin"""
        import struct
        path = self.sfm_dir / 'points3D.bin'
        points = []
        with open(path, 'rb') as f:
            num_points = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_points):
                f.read(8)  # point3D_id
                xyz = struct.unpack('<3d', f.read(24))
                points.append(xyz)
                f.read(3 + 8)  # RGB + error
                track_len = struct.unpack('<Q', f.read(8))[0]
                f.read(8 * track_len)
        return np.array(points, dtype=np.float32)


# Test the module
if __name__ == '__main__':
    test_image_dir = '/content/drive/MyDrive/heritage_reconstruction_shared/data/tanks_and_temples/your_scene/images_sr'
    
    if os.path.exists(test_image_dir):
        sfm = DenseSfMInitializer(test_image_dir, '/tmp/sfm_test')
        result = sfm.run()
        print(f"✅ Test passed: {len(result['points'])} points")
    else:
        print("⚠️ No test images found. Create a test folder with 10-20 images first.")