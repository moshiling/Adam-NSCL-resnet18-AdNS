import time

import torch

from utils.subspace_utils import compute_candidate_nullspace


def _safe_norm(matrix):
    norm = torch.norm(matrix)
    if not torch.isfinite(norm) or norm <= 0:
        return None
    return norm


def _summarize_tensor(values):
    if values.numel() == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0}
    return {
        'min': float(values.min().detach().cpu().item()),
        'max': float(values.max().detach().cpu().item()),
        'mean': float(values.mean().detach().cpu().item()),
    }


def _two_stage_scales(eigen_values, null_mask, tau1, tau2, rho_t=1.0):
    scales = torch.empty_like(eigen_values)
    high_scale = 1.0 / (1.0 + max(float(tau1), 0.0))
    low_scale = max(float(tau2), 0.0) / (1.0 + max(float(tau2), 0.0))
    scales[~null_mask] = high_scale * float(rho_t)
    scales[null_mask] = low_scale
    return torch.clamp(scales, min=0.0)


def _projector_from_basis(basis, dim, device, dtype, normalize=True):
    if basis is None or basis.numel() == 0:
        matrix = torch.eye(dim, device=device, dtype=dtype)
    else:
        matrix = torch.mm(basis, basis.transpose(0, 1))
    if normalize:
        norm = _safe_norm(matrix)
        if norm is not None:
            matrix = matrix / norm
    return matrix


def build_nscl_projector(eigen_vectors, eigen_values, config):
    start = time.time()
    basis, null_mask, null_stats = compute_candidate_nullspace(
        eigen_vectors, eigen_values, config['thres'])
    projector = _projector_from_basis(
        basis,
        dim=eigen_vectors.shape[0],
        device=eigen_vectors.device,
        dtype=eigen_vectors.dtype,
        normalize=config.get('norm_projection', True),
    )
    stats = {
        'mode': 'nscl',
        'singular': _summarize_tensor(eigen_values),
        'scale': {'min': 0.0, 'max': 1.0, 'mean': float(null_mask.float().mean().detach().cpu().item())},
        'candidate_nullspace': null_stats,
        'elapsed_sec': time.time() - start,
    }
    return projector, basis, stats


def build_sfcl_projector(eigen_vectors, eigen_values, config):
    start = time.time()
    basis, null_mask, null_stats = compute_candidate_nullspace(
        eigen_vectors, eigen_values, config['thres'])
    high_scale = 1.0 / (1.0 + max(float(config.get('sfcl_tau1', 10.0)), 0.0))
    rho_t = float(config.get('rho_t', 1.0))
    scales = _two_stage_scales(
        eigen_values,
        null_mask,
        tau1=config.get('sfcl_tau1', 10.0),
        tau2=config.get('sfcl_tau2', 10.0),
        rho_t=rho_t,
    )
    projector = torch.mm(eigen_vectors * scales.unsqueeze(0), eigen_vectors.transpose(0, 1))
    projector_norm = _safe_norm(projector)
    if config.get('norm_projection', True):
        if projector_norm is not None:
            projector = projector / projector_norm
    stats = {
        'mode': 'sfcl',
        'singular': _summarize_tensor(eigen_values),
        'scale': _summarize_tensor(scales),
        'base_scale': _summarize_tensor(scales),
        'rho_t': rho_t,
        'high_scale_before_rho': high_scale,
        'high_scale_after_rho': high_scale * rho_t,
        'projector_norm': 0.0 if projector_norm is None else float(projector_norm.detach().cpu().item()),
        'candidate_nullspace': null_stats,
        'elapsed_sec': time.time() - start,
    }
    return projector, basis, stats


def build_sfcl_adns_projector(eigen_vectors, eigen_values, config, shared_basis=None):
    start = time.time()
    _, basis, stats = build_sfcl_projector(eigen_vectors, eigen_values, config)
    null_mask = compute_candidate_nullspace(eigen_vectors, eigen_values, config['thres'])[1]
    base_scales = _two_stage_scales(
        eigen_values,
        null_mask,
        tau1=config.get('sfcl_tau1', 10.0),
        tau2=config.get('sfcl_tau2', 10.0),
        rho_t=config.get('rho_t', 1.0),
    )

    safe_scores = torch.zeros_like(eigen_values)
    if shared_basis is not None and shared_basis.numel() > 0:
        overlap = torch.mm(shared_basis.transpose(0, 1), eigen_vectors)
        safe_scores = torch.clamp((overlap ** 2).sum(dim=0), min=0.0, max=1.0)

    boosts = config.get('risk_shrink', 1.0) + (
        config.get('safe_boost', 1.0) - config.get('risk_shrink', 1.0)
    ) * safe_scores
    final_scales = torch.clamp(base_scales * boosts, min=0.0)
    projector = torch.mm(eigen_vectors * final_scales.unsqueeze(0), eigen_vectors.transpose(0, 1))
    projector_norm = _safe_norm(projector)
    if config.get('norm_projection', True):
        if projector_norm is not None:
            projector = projector / projector_norm

    stats.update({
        'mode': 'sfcl_adns',
        'scale': _summarize_tensor(final_scales),
        'base_scale': _summarize_tensor(base_scales),
        'safe_score': _summarize_tensor(safe_scores),
        'boost': _summarize_tensor(boosts),
        'final_scale': _summarize_tensor(final_scales),
        'projector_norm': 0.0 if projector_norm is None else float(projector_norm.detach().cpu().item()),
        'elapsed_sec': time.time() - start,
    })
    return projector, basis, stats
