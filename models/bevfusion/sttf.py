"""Spatial Temporal Feature Transform (STTF) module.

Warps BEV features from cooperating agents into the ego agent's coordinate
frame using differentiable affine warping (F.affine_grid + F.grid_sample).

Adapted from CoBEVT:
    CoBEVT/opv2v/opencood/models/corpbevt.py
    CoBEVT/opv2v/opencood/models/sub_modules/torch_transformation_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_discretized_transformation_matrix(matrix, discrete_ratio,
                                          downsample_rate):
    """Extract 2D affine (2x3) from 4x4 transformation matrix.

    Selects rows [0,1] and cols [0,1,3] from the 4x4 matrix, then
    normalizes the translation component by the BEV pixel size.

    Args:
        matrix: (B, L, 4, 4) transformation matrices.
        discrete_ratio: Meters per BEV pixel.
        downsample_rate: Additional downsample factor.

    Returns:
        (B, L, 2, 3) discretized 2D affine matrices.
    """
    # Select 2D rows and cols: rows [0,1], cols [0,1,3]
    matrix = matrix[:, :, [0, 1], :][:, :, :, [0, 1, 3]]
    # Normalize translation by pixel size
    matrix[:, :, :, -1] = (
        matrix[:, :, :, -1] / (discrete_ratio * downsample_rate))
    return matrix.float()


def _torch_inverse_cast(input):
    """Safe inverse that casts to float32 if needed."""
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)


def normal_transform_pixel(height, width, device, dtype, eps=1e-14):
    """Compute normalization matrix from pixel coords to [-1, 1].

    Returns:
        (1, 3, 3) normalization matrix.
    """
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device, dtype=dtype)
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    return tr_mat.unsqueeze(0)


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst=None):
    """Normalize a pixel-space homography to [-1, 1] grid coords."""
    if dsize_dst is None:
        dsize_dst = dsize_src
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    device = dst_pix_trans_src_pix.device
    dtype = dst_pix_trans_src_pix.dtype

    src_norm_trans_src_pix = normal_transform_pixel(
        src_h, src_w, device, dtype).to(dst_pix_trans_src_pix)
    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(
        dst_h, dst_w, device, dtype).to(dst_pix_trans_src_pix)

    dst_norm_trans_src_norm = (
        dst_norm_trans_dst_pix
        @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm))
    return dst_norm_trans_src_norm


def get_rotation_matrix2d(M, dsize):
    """Build rotation-around-center affine from a 2x3 matrix.

    The rotation/scale part of M is applied around the center of the image,
    then the translation part is added.

    Args:
        M: (B, 2, 3) affine matrix with rotation in [:2,:2] and
           translation in [:, 2].
        dsize: (H, W) spatial dimensions.

    Returns:
        (B, 2, 3) affine matrix for center-based rotation + translation.
    """
    H, W = dsize
    B = M.shape[0]
    center = torch.tensor(
        [W / 2, H / 2], dtype=M.dtype, device=M.device).unsqueeze(0)

    eye = torch.eye(3, device=M.device, dtype=M.dtype).unsqueeze(0).repeat(
        B, 1, 1)

    shift_m = eye.clone()
    shift_m[:, :2, 2] = center

    shift_m_inv = eye.clone()
    shift_m_inv[:, :2, 2] = -center

    rotat_m = eye.clone()
    rotat_m[:, :2, :2] = M[:, :2, :2]

    affine_m = shift_m @ rotat_m @ shift_m_inv
    return affine_m[:, :2, :]


def get_transformation_matrix(M, dsize):
    """Get full transformation matrix (rotation around center + translation).

    Args:
        M: (B, 2, 3) with rotation in [:2,:2] and translation in [:, 2].
        dsize: (H, W) spatial dimensions.

    Returns:
        (B, 2, 3) full affine transformation matrix.
    """
    T = get_rotation_matrix2d(M, dsize)
    T[..., 2] += M[..., 2]
    return T


def convert_affinematrix_to_homography(A):
    """Pad a 2x3 affine to 3x3 homography."""
    H = F.pad(A, [0, 0, 0, 1], 'constant', value=0.0)
    H[..., -1, -1] += 1.0
    return H


def warp_affine(src, M, dsize, mode='bilinear', padding_mode='zeros',
                align_corners=True):
    """Differentiable affine warp using grid_sample.

    Args:
        src: (B, C, H, W) source features.
        M: (B, 2, 3) affine transformation matrix.
        dsize: (H, W) output spatial size.
        mode: Interpolation mode for grid_sample.
        padding_mode: Padding mode for grid_sample.
        align_corners: Whether to align corners in grid_sample.

    Returns:
        (B, C, H, W) warped features.
    """
    B, C, H, W = src.size()
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, (H, W), dsize)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners)
    return F.grid_sample(
        src.float(), grid.float(),
        align_corners=align_corners, mode=mode,
        padding_mode=padding_mode)


class STTF(nn.Module):
    """Spatial Temporal Feature Transform.

    Warps BEV features from each cooperating agent into the ego agent's
    coordinate frame using differentiable affine warping.

    Args:
        discrete_ratio: Meters per BEV pixel (e.g. 0.8 for 144m / 180 pixels).
        downsample_rate: Additional downsample factor (typically 1).
    """

    def __init__(self, discrete_ratio, downsample_rate):
        super().__init__()
        self.discrete_ratio = discrete_ratio
        self.downsample_rate = downsample_rate

    def forward(self, x, spatial_correction_matrix):
        """Warp BEV features to ego coordinate frame.

        Args:
            x: (B, L, C, H, W) BEV features for all agents.
            spatial_correction_matrix: (B, L, 4, 4) cav-to-ego transforms.

        Returns:
            (B, L, C, H, W) warped BEV features in ego frame.
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix.clone(),
            self.discrete_ratio,
            self.downsample_rate)

        # Transpose and flip to match CoBEVT's coordinate convention
        x = rearrange(x, 'b l c h w -> b l c w h')
        x = torch.flip(x, dims=(4,))

        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix.reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(
            x.reshape(-1, C, H, W), T, (H, W))
        cav_features = cav_features.reshape(B, L, C, H, W)

        # Flip and transpose back
        x = torch.flip(cav_features, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l c h w')

        return x
