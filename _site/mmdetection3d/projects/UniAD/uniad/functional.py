"""Pure PyTorch utility functions ported from UniAD."""
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def bivariate_gaussian_nll_loss(pred, target, mask=None, reduction='mean'):
    """Bivariate Gaussian negative log-likelihood loss.

    This is the actual loss UniAD uses for trajectory regression,
    supervising all 5 output dimensions (mu_x, mu_y, sigma_x, sigma_y, rho).

    Args:
        pred: (..., 5) with [mu_x, mu_y, sigma_x, sigma_y, rho]
              sigma_x/y already exp-activated, rho already tanh-activated.
        target: (..., 2) ground truth [x, y].
        mask: (...,) or (..., 1) validity mask.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss or per-element loss if reduction='none'.
    """
    mu_x = pred[..., 0]
    mu_y = pred[..., 1]
    sig_x = pred[..., 2].clamp(min=1e-4)
    sig_y = pred[..., 3].clamp(min=1e-4)
    rho = pred[..., 4].clamp(min=-0.999, max=0.999)

    dx = target[..., 0] - mu_x
    dy = target[..., 1] - mu_y

    one_minus_rho2 = (1 - rho ** 2).clamp(min=1e-4)

    nll = (0.5 / one_minus_rho2 * (
        (dx / sig_x) ** 2 + (dy / sig_y) ** 2
        - 2 * rho * dx * dy / (sig_x * sig_y))
        + torch.log(sig_x) + torch.log(sig_y)
        + 0.5 * torch.log(one_minus_rho2)
        + math.log(2 * math.pi))

    if mask is not None:
        if mask.dim() == nll.dim() + 1:
            mask = mask.squeeze(-1)
        nll = nll * mask

    if reduction == 'mean':
        if mask is not None:
            return nll.sum() / mask.sum().clamp(min=1)
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    return nll


def bivariate_gaussian_activation(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    return torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)


def norm_points(pos, pc_range):
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def anchor_coordinate_transform(anchors, bbox_results, with_mask=True):
    """Transform anchors from global to agent-centric coordinates."""
    if bbox_results is None:
        return anchors

    B, A, P, T, _ = anchors.shape
    bboxes = bbox_results[0]  # (A, 10)
    cx, cy = bboxes[:, 0], bboxes[:, 1]
    sin_yaw, cos_yaw = bboxes[:, 6], bboxes[:, 7]

    # Rotate + translate
    anchors_out = anchors.clone()
    x = anchors[..., 0]
    y = anchors[..., 1]
    anchors_out[..., 0] = cos_yaw[None, :, None, None] * x - sin_yaw[None, :, None, None] * y + cx[None, :, None, None]
    anchors_out[..., 1] = sin_yaw[None, :, None, None] * x + cos_yaw[None, :, None, None] * y + cy[None, :, None, None]

    return anchors_out


def nonlinear_smoother(trajectory, sigma=1.0):
    """Apply Gaussian smoothing to trajectories."""
    if trajectory.shape[-2] <= 2:
        return trajectory
    T = trajectory.shape[-2]
    kernel_size = T
    x = torch.arange(kernel_size, dtype=torch.float32, device=trajectory.device)
    x = x - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Apply 1D convolution for smoothing
    orig_shape = trajectory.shape
    traj_flat = trajectory.reshape(-1, T, trajectory.shape[-1])
    traj_smooth = torch.zeros_like(traj_flat)
    for d in range(traj_flat.shape[-1]):
        padded = torch.nn.functional.pad(
            traj_flat[:, :, d].unsqueeze(1),
            (kernel_size // 2, kernel_size // 2),
            mode='replicate'
        )
        traj_smooth[:, :, d] = torch.nn.functional.conv1d(
            padded, kernel.view(1, 1, -1)
        ).squeeze(1)

    return traj_smooth.reshape(orig_shape)
