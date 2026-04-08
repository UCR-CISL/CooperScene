"""Custom data transforms for UniAD pipeline."""
import os.path as osp

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadBEVMap(BaseTransform):
    """Load BEV map PNG images (lane + static) and stack into a tensor.

    Expects the data_info dict to contain 'bev_map_paths' with paths to
    PNG files. Loads them, resizes to map_size, normalizes to [0,1], and
    stacks into a single array.

    Args:
        map_types (list[str]): Map types to load. Default: ['bev_lane', 'bev_static'].
        map_size (int): Resize maps to this size. Default: 256.
    """

    def __init__(self, map_types=None, map_size=256):
        if map_types is None:
            map_types = ['bev_lane', 'bev_static']
        self.map_types = map_types
        self.map_size = map_size

    def transform(self, results: dict) -> dict:
        """Load and stack BEV map images.

        Adds 'bev_map' key: (C, H, W) float32 array in [0, 1].
        For 2 RGB maps: C=6, H=W=map_size.
        """
        from PIL import Image

        bev_map_paths = results.get('bev_map_paths', {})
        channels = []

        for map_type in self.map_types:
            path = bev_map_paths.get(map_type, '')
            if path and osp.exists(path):
                img = Image.open(path).convert('RGB')
                if img.size != (self.map_size, self.map_size):
                    img = img.resize(
                        (self.map_size, self.map_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                channels.append(arr.transpose(2, 0, 1))  # (3, H, W)
            else:
                # Missing map: fill with zeros
                channels.append(
                    np.zeros((3, self.map_size, self.map_size),
                             dtype=np.float32))

        # Stack: (C, H, W) where C = len(map_types) * 3
        results['bev_map'] = np.concatenate(channels, axis=0)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'map_types={self.map_types}, map_size={self.map_size})')
