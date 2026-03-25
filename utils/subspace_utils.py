import math
import time

import torch


def _empty_basis(device=None, dtype=None, dim=0):
    return torch.empty(dim, 0, device=device, dtype=dtype)


def orthonormalize_basis(basis, eps=1e-8):
    if basis is None:
        return None
    if basis.numel() == 0:
        return basis
    q, r = torch.linalg.qr(basis, mode='reduced')
    if r.numel() == 0:
        return basis[:, :0]
    keep = torch.abs(torch.diag(r)) > eps
    if keep.ndim == 0:
        keep = keep.unsqueeze(0)
    return q[:, keep]


def compute_candidate_nullspace(eigen_vectors, eigen_values, thres):
    device = eigen_vectors.device
    dtype = eigen_vectors.dtype
    dim = eigen_vectors.shape[0]
    if eigen_values.numel() == 0:
        return _empty_basis(device=device, dtype=dtype, dim=dim), torch.zeros(0, dtype=torch.bool, device=device), {
            'threshold_value': 0.0,
            'candidate_dim': 0,
            'full_dim': dim,
        }

    min_value = torch.clamp(eigen_values[-1], min=0.0)
    threshold_value = min_value * thres
    mask = eigen_values <= threshold_value
    if mask.sum() == 0:
        mask[-1] = True
    basis = eigen_vectors[:, mask]
    stats = {
        'threshold_value': float(threshold_value.detach().cpu().item()),
        'candidate_dim': int(mask.sum().item()),
        'full_dim': int(dim),
        'min_singular': float(eigen_values[-1].detach().cpu().item()),
        'max_singular': float(eigen_values[0].detach().cpu().item()),
    }
    return basis, mask, stats


def _rank_target(p, q, mode, ratio):
    if mode == 'max':
        base = max(p, q)
    elif mode == 'min':
        base = min(p, q)
    else:
        base = 0.5 * (p + q)
    return max(1, int(math.ceil(base * ratio)))


def _empty_shared_stats(p, q, elapsed_sec, mode_name):
    return {
        'mode': mode_name,
        'p': int(p),
        'q': int(q),
        'k': 0,
        'elapsed_sec': float(elapsed_sec),
    }


def compute_shared_core_subspace(
    u_pre,
    u_cur,
    rank_mode='avg',
    rank_ratio=0.9,
    overlap_threshold=0.5,
):
    start = time.time()
    if u_pre is None or u_pre.numel() == 0:
        result = orthonormalize_basis(u_cur) if u_cur is not None else None
        stats = _empty_shared_stats(0, 0 if u_cur is None else u_cur.shape[1], time.time() - start, 'overlap_core')
        if result is not None:
            stats['k'] = int(result.shape[1])
        return result, stats
    if u_cur is None or u_cur.numel() == 0:
        result = orthonormalize_basis(u_pre)
        stats = _empty_shared_stats(u_pre.shape[1], 0, time.time() - start, 'overlap_core')
        if result is not None:
            stats['k'] = int(result.shape[1])
        return result, stats

    u_pre = orthonormalize_basis(u_pre)
    u_cur = orthonormalize_basis(u_cur)
    p = int(u_pre.shape[1])
    q = int(u_cur.shape[1])
    if u_pre.numel() == 0 or u_cur.numel() == 0:
        return _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=u_cur.shape[0]), _empty_shared_stats(p, q, time.time() - start, 'overlap_core')

    overlap = torch.mm(u_pre.transpose(0, 1), u_cur)
    left, singular_values, right = torch.linalg.svd(overlap, full_matrices=False)
    max_rank = min(left.shape[1], right.shape[0], singular_values.numel())
    target_rank = min(max_rank, _rank_target(p, q, rank_mode, rank_ratio))
    keep_mask = singular_values >= float(overlap_threshold)
    keep_count = int(keep_mask.sum().item()) if keep_mask.numel() > 0 else 0
    if keep_count == 0 and target_rank > 0 and singular_values.numel() > 0:
        keep_count = target_rank
    keep_count = min(max_rank, keep_count if keep_count > 0 else target_rank)

    if keep_count <= 0:
        shared = _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=u_cur.shape[0])
    else:
        # Use overlap-aligned canonical directions from both subspaces, then average and orthonormalize.
        pre_core = torch.mm(u_pre, left[:, :keep_count])
        cur_core = torch.mm(u_cur, right.transpose(0, 1)[:, :keep_count])
        shared = orthonormalize_basis(0.5 * (pre_core + cur_core))
        if shared is None or shared.numel() == 0:
            shared = orthonormalize_basis(pre_core)
        if shared is None or shared.numel() == 0:
            shared = orthonormalize_basis(cur_core)
        if shared is None:
            shared = _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=u_cur.shape[0])

    angle_stats = maybe_compute_principal_angles(u_pre, u_cur)
    stats = {
        'mode': 'overlap_core',
        'p': p,
        'q': q,
        'k': int(shared.shape[1]),
        'target_rank': int(target_rank),
        'threshold_keep': int(keep_mask.sum().item()) if singular_values.numel() > 0 else 0,
        'overlap_threshold': float(overlap_threshold),
        'overlap_singular': _summarize_values(singular_values),
        'elapsed_sec': time.time() - start,
    }
    if angle_stats is not None:
        stats['principal_angles'] = angle_stats
    return shared, stats


def _summarize_values(values):
    if values is None or values.numel() == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0}
    return {
        'min': float(values.min().detach().cpu().item()),
        'max': float(values.max().detach().cpu().item()),
        'mean': float(values.mean().detach().cpu().item()),
    }


def _compute_union_lowrank_subspace(u_pre, u_cur, mode='avg', ratio=0.9):
    start = time.time()
    if u_cur is None:
        u_cur = None
    if u_pre is None or u_pre.numel() == 0:
        result = orthonormalize_basis(u_cur) if u_cur is not None else None
        stats = {
            'p': 0,
            'q': 0 if u_cur is None else int(u_cur.shape[1]),
            'k': 0 if result is None else int(result.shape[1]),
            'elapsed_sec': time.time() - start,
        }
        return result, stats
    if u_cur is None or u_cur.numel() == 0:
        result = orthonormalize_basis(u_pre)
        stats = {
            'p': int(u_pre.shape[1]),
            'q': 0,
            'k': 0 if result is None else int(result.shape[1]),
            'elapsed_sec': time.time() - start,
        }
        return result, stats

    u_pre = orthonormalize_basis(u_pre)
    u_cur = orthonormalize_basis(u_cur)
    dim = u_cur.shape[0]
    u_cat = torch.cat([u_pre, u_cur], dim=1)
    if u_cat.numel() == 0:
        return _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=dim), {
            'p': 0,
            'q': 0,
            'k': 0,
            'elapsed_sec': time.time() - start,
        }

    u_cat = orthonormalize_basis(u_cat)
    if u_cat.numel() == 0:
        return _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=dim), {
            'p': int(u_pre.shape[1]),
            'q': int(u_cur.shape[1]),
            'k': 0,
            'elapsed_sec': time.time() - start,
        }

    left, singular_values, _ = torch.linalg.svd(u_cat, full_matrices=False)
    target_k = min(left.shape[1], _rank_target(u_pre.shape[1], u_cur.shape[1], mode, ratio))
    if singular_values.numel() > 0:
        positive = singular_values > 1e-8
        target_k = min(target_k, int(positive.sum().item())) if positive.any() else 0
    shared = left[:, :target_k] if target_k > 0 else _empty_basis(device=u_cur.device, dtype=u_cur.dtype, dim=dim)
    shared = orthonormalize_basis(shared)
    stats = {
        'mode': 'union_lowrank',
        'p': int(u_pre.shape[1]),
        'q': int(u_cur.shape[1]),
        'k': int(shared.shape[1]),
        'cat_rank': int(u_cat.shape[1]),
        'elapsed_sec': time.time() - start,
    }
    return shared, stats


def compute_shared_lowrank_subspace(
    u_pre,
    u_cur,
    mode='avg',
    ratio=0.9,
    shared_subspace_mode='overlap_core',
    overlap_threshold=0.5,
):
    if shared_subspace_mode == 'union_lowrank':
        return _compute_union_lowrank_subspace(u_pre, u_cur, mode=mode, ratio=ratio)
    return compute_shared_core_subspace(
        u_pre,
        u_cur,
        rank_mode=mode,
        rank_ratio=ratio,
        overlap_threshold=overlap_threshold,
    )


def maybe_compute_principal_angles(u_a, u_b):
    if u_a is None or u_b is None:
        return None
    if u_a.numel() == 0 or u_b.numel() == 0:
        return None
    u_a = orthonormalize_basis(u_a)
    u_b = orthonormalize_basis(u_b)
    if u_a.numel() == 0 or u_b.numel() == 0:
        return None
    sigma = torch.linalg.svdvals(torch.mm(u_a.transpose(0, 1), u_b))
    sigma = torch.clamp(sigma, -1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(sigma))
    return {
        'min_deg': float(angles.min().detach().cpu().item()),
        'max_deg': float(angles.max().detach().cpu().item()),
        'mean_deg': float(angles.mean().detach().cpu().item()),
        'overlap_mean': float(sigma.mean().detach().cpu().item()),
    }
