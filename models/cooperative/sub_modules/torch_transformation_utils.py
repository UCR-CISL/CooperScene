"""Pose-aware feature warping utilities (SE2 affine, ROI masks).
"""
import torch
import torch.nn.functional as F
import numpy as np


def get_roi_and_cav_mask(shape, cav_mask, spatial_correction_matrix,
                         discrete_ratio, downsample_rate):
    B, L, H, W, C = shape
    C = 1
    dist_correction_matrix = get_discretized_transformation_matrix(
        spatial_correction_matrix, discrete_ratio, downsample_rate)
    T = get_transformation_matrix(
        dist_correction_matrix.reshape(-1, 2, 3), (H, W))
    roi_mask = get_rotated_roi((B, L, C, H, W), T)
    com_mask = combine_roi_and_cav_mask(roi_mask, cav_mask)
    com_mask = com_mask.permute(0, 3, 4, 2, 1)
    return com_mask


def combine_roi_and_cav_mask(roi_mask, cav_mask):
    cav_mask = cav_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    cav_mask = cav_mask.expand(roi_mask.shape)
    com_mask = roi_mask * cav_mask
    return com_mask


def get_rotated_roi(shape, correction_matrix):
    B, L, C, H, W = shape
    x = torch.ones((B, L, 1, H, W)).to(correction_matrix.dtype).to(
        correction_matrix.device)
    roi_mask = warp_affine(x.reshape(-1, 1, H, W), correction_matrix,
                           dsize=(H, W), mode="nearest")
    roi_mask = torch.repeat_interleave(roi_mask, C, dim=1).reshape(B, L, C, H, W)
    return roi_mask


def get_discretized_transformation_matrix(matrix, discrete_ratio,
                                          downsample_rate):
    matrix = matrix[:, :, [0, 1], :][:, :, :, [0, 1, 3]]
    matrix[:, :, :, -1] = matrix[:, :, :, -1] \
                          / (discrete_ratio * downsample_rate)
    return matrix.type(dtype=torch.float)


def _torch_inverse_cast(input):
    dtype = input.dtype
    device = input.device
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    input_cpu = input.cpu().to(dtype)
    out = torch.inverse(input_cpu)
    out = out.to(device=device, dtype=input.dtype)
    return out


def normal_transform_pixel(height, width, device, dtype, eps=1e-14):
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device, dtype=dtype)
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    return tr_mat.unsqueeze(0)


def eye_like(n, B, device, dtype):
    identity = torch.eye(n, device=device, dtype=dtype)
    return identity[None].repeat(B, 1, 1)


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst=None):
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
    dst_norm_trans_src_norm = dst_norm_trans_dst_pix @ (
            dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def get_rotation_matrix2d(M, dsize):
    H, W = dsize
    B = M.shape[0]
    center = torch.Tensor([W / 2, H / 2]).to(M.dtype).to(M.device).unsqueeze(0)
    shift_m = eye_like(3, B, M.device, M.dtype)
    shift_m[:, :2, 2] = center
    shift_m_inv = eye_like(3, B, M.device, M.dtype)
    shift_m_inv[:, :2, 2] = -center
    rotat_m = eye_like(3, B, M.device, M.dtype)
    rotat_m[:, :2, :2] = M[:, :2, :2]
    affine_m = shift_m @ rotat_m @ shift_m_inv
    return affine_m[:, :2, :]


def get_transformation_matrix(M, dsize):
    T = get_rotation_matrix2d(M, dsize)
    T[..., 2] += M[..., 2]
    return T


def convert_affinematrix_to_homography(A):
    H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def warp_affine(src, M, dsize, mode='bilinear', padding_mode='zeros',
                align_corners=True):
    B, C, H, W = src.size()
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, (H, W), dsize)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners)
    return F.grid_sample(
        src.half() if grid.dtype == torch.half else src,
        grid, align_corners=align_corners, mode=mode,
        padding_mode=padding_mode)
