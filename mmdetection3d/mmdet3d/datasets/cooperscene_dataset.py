"""CooperScene Dataset for 3D Object Detection.

Extends OPV2VDataset. The only difference is that ego_pose is stored
as a 4x4 matrix (list of lists) instead of a 6-element list.
For single-agent LiDAR training, ego_pose is just metadata and not
used in computation, so this class is a thin wrapper.
"""

from typing import Callable, List, Optional, Union

from mmdet3d.registry import DATASETS
from .opv2v_dataset import OPV2VDataset


@DATASETS.register_module()
class CooperSceneDataset(OPV2VDataset):
    """CooperScene single-agent dataset.

    Identical to OPV2VDataset. ego_pose is stored as a 4x4 matrix
    but only used as metadata, so no code changes are needed.
    """

    METAINFO = {
        'classes': ('vehicle',),
        'palette': [(255, 158, 0)],
    }
