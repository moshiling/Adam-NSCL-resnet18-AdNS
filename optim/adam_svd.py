import math
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer

from optim.projection_builder import (
    build_nscl_projector,
    build_sfcl_adns_projector,
    build_sfcl_projector,
)
from utils.subspace_utils import (
    compute_shared_lowrank_subspace,
    maybe_compute_principal_angles,
    orthonormalize_basis,
)


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        svd=False,
        thres=1.001,
        weight_decay=0,
        amsgrad=False,
        projection_mode='nscl',
        sfcl_tau1=10.0,
        sfcl_tau2=10.0,
        sfcl_norm_projection=True,
        use_shared_lowrank=False,
        shared_rank_mode='avg',
        shared_rank_ratio=0.9,
        shared_subspace_mode='overlap_core',
        shared_overlap_threshold=0.5,
        safe_boost=1.25,
        risk_shrink=0.75,
        use_task_strength=False,
        alpha_t=1.0,
        rho_t=1.0,
        use_rho_t=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            svd=svd,
            thres=thres,
            projection_mode=projection_mode,
            sfcl_tau1=sfcl_tau1,
            sfcl_tau2=sfcl_tau2,
            sfcl_norm_projection=sfcl_norm_projection,
            use_shared_lowrank=use_shared_lowrank,
            shared_rank_mode=shared_rank_mode,
            shared_rank_ratio=shared_rank_ratio,
            shared_subspace_mode=shared_subspace_mode,
            shared_overlap_threshold=shared_overlap_threshold,
            safe_boost=safe_boost,
            risk_shrink=risk_shrink,
            use_task_strength=use_task_strength,
            alpha_t=alpha_t,
            rho_t=rho_t,
            use_rho_t=use_rho_t,
        )
        super(Adam, self).__init__(params, defaults)
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.shared_subspaces = {}
        self.current_task_context = {'task_index': 1, 'total_tasks': None, 'alpha_t': 1.0, 'rho_t': 1.0}
        self.param_to_name = {}
        self.name_to_param = {}
        self.latest_build_stats = {}
        self._task_stat_buffer = defaultdict(lambda: {'raw_norm_sum': 0.0, 'proj_norm_sum': 0.0, 'final_norm_sum': 0.0, 'steps': 0})

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def set_param_metadata(self, param_to_name):
        self.param_to_name = dict(param_to_name)
        self.name_to_param = {name: param for param, name in self.param_to_name.items()}

    def set_task_context(self, task_index=1, total_tasks=None, alpha_t=1.0, rho_t=1.0):
        self.current_task_context = {
            'task_index': int(task_index),
            'total_tasks': total_tasks,
            'alpha_t': float(alpha_t),
            'rho_t': float(rho_t),
        }
        for group in self.param_groups:
            group['alpha_t'] = float(alpha_t)
            group['rho_t'] = float(rho_t)

    def reset_task_stats(self):
        self._task_stat_buffer = defaultdict(lambda: {'raw_norm_sum': 0.0, 'proj_norm_sum': 0.0, 'final_norm_sum': 0.0, 'steps': 0})

    def get_task_stats(self):
        summary = {}
        for name, stats in self._task_stat_buffer.items():
            steps = max(1, stats['steps'])
            summary[name] = {
                'avg_raw_grad_norm': stats['raw_norm_sum'] / steps,
                'avg_projected_grad_norm': stats['proj_norm_sum'] / steps,
                'avg_final_grad_norm': stats['final_norm_sum'] / steps,
                'steps': stats['steps'],
            }
        return summary

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                update_projected = update
                if svd and p in self.transforms and torch.is_tensor(self.transforms[p]):
                    projector = self.transforms[p]
                    if len(update.shape) == 4:
                        update_projected = torch.mm(update.view(update.size(0), -1), projector).view_as(update)
                    else:
                        update_projected = torch.mm(update, projector)
                    if group.get('use_task_strength', False):
                        alpha_t = float(group.get('alpha_t', 1.0))
                        update_final = (1.0 - alpha_t) * update + alpha_t * update_projected
                    else:
                        update_final = update_projected
                else:
                    update_final = update

                param_name = self.param_to_name.get(p)
                if param_name is not None:
                    raw_norm = float(torch.norm(update).detach().cpu().item())
                    proj_norm = float(torch.norm(update_projected).detach().cpu().item())
                    final_norm = float(torch.norm(update_final).detach().cpu().item())
                    buf = self._task_stat_buffer[param_name]
                    buf['raw_norm_sum'] += raw_norm
                    buf['proj_norm_sum'] += proj_norm
                    buf['final_norm_sum'] += final_norm
                    buf['steps'] += 1

                p.data.add_(update_final)
        return loss

    def _projector_config(self, group):
        return {
            'thres': float(group['thres']),
            'sfcl_tau1': float(group.get('sfcl_tau1', 10.0)),
            'sfcl_tau2': float(group.get('sfcl_tau2', 10.0)),
            'norm_projection': bool(group.get('sfcl_norm_projection', True)),
            'safe_boost': float(group.get('safe_boost', 1.25)),
            'risk_shrink': float(group.get('risk_shrink', 0.75)),
            'rho_t': float(group.get('rho_t', 1.0)),
        }

    def get_transforms(self):
        self.latest_build_stats = {}
        for group in self.param_groups:
            if not group['svd']:
                continue
            for p in group['params']:
                if p not in self.eigens or len(self.eigens[p]) == 0:
                    continue

                param_name = self.param_to_name.get(p, 'unknown')
                eigen_values = self.eigens[p]['eigen_value']
                eigen_vectors = self.eigens[p]['eigen_vector']
                projection_mode = group.get('projection_mode', 'nscl')
                projector_cfg = self._projector_config(group)

                shared_stats = None
                principal_angle_stats = None
                current_basis = None
                shared_basis = self.shared_subspaces.get(param_name)
                if projection_mode == 'nscl':
                    projector, current_basis, build_stats = build_nscl_projector(eigen_vectors, eigen_values, projector_cfg)
                elif projection_mode == 'sfcl':
                    projector, current_basis, build_stats = build_sfcl_projector(eigen_vectors, eigen_values, projector_cfg)
                else:
                    if group.get('use_shared_lowrank', False):
                        current_basis = build_sfcl_projector(eigen_vectors, eigen_values, projector_cfg)[1]
                        shared_basis, shared_stats = compute_shared_lowrank_subspace(
                            self.shared_subspaces.get(param_name),
                            current_basis,
                            mode=group.get('shared_rank_mode', 'avg'),
                            ratio=float(group.get('shared_rank_ratio', 0.9)),
                            shared_subspace_mode=group.get('shared_subspace_mode', 'overlap_core'),
                            overlap_threshold=float(group.get('shared_overlap_threshold', 0.5)),
                        )
                        if shared_basis is not None:
                            shared_basis = orthonormalize_basis(shared_basis)
                            self.shared_subspaces[param_name] = shared_basis.detach()
                        principal_angle_stats = maybe_compute_principal_angles(shared_basis, current_basis)
                    projector, current_basis, build_stats = build_sfcl_adns_projector(
                        eigen_vectors,
                        eigen_values,
                        projector_cfg,
                        shared_basis=shared_basis if group.get('use_shared_lowrank', False) else None,
                    )

                if projection_mode in ['nscl', 'sfcl'] and current_basis is not None:
                    self.shared_subspaces.setdefault(param_name, current_basis.detach())

                self.transforms[p] = projector.detach()
                self.latest_build_stats[param_name] = build_stats
                if shared_stats is not None:
                    self.latest_build_stats[param_name]['shared_lowrank'] = shared_stats
                if principal_angle_stats is not None:
                    self.latest_build_stats[param_name]['principal_angles'] = principal_angle_stats

    def get_eigens(self, fea_in):
        for group in self.param_groups:
            if not group['svd']:
                continue
            for p in group['params']:
                if p not in fea_in:
                    continue
                cov = fea_in[p]
                if cov.numel() == 0:
                    continue
                eigen = self.eigens[p]
                u, s, _ = torch.linalg.svd(cov, full_matrices=False)
                eigen['eigen_value'] = s
                eigen['eigen_vector'] = u

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1

        if group['weight_decay'] != 0:
            grad = grad.add(p.data, alpha=group['weight_decay'])

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        update = -step_size * exp_avg / denom
        return update

    def serialize_projection_state(self):
        def cpu_tensor(tensor):
            return None if tensor is None else tensor.detach().cpu()

        eigens = {}
        transforms = {}
        for param, name in self.param_to_name.items():
            if param in self.eigens and len(self.eigens[param]) > 0:
                eigens[name] = {
                    'eigen_value': cpu_tensor(self.eigens[param]['eigen_value']),
                    'eigen_vector': cpu_tensor(self.eigens[param]['eigen_vector']),
                }
            if param in self.transforms and torch.is_tensor(self.transforms[param]):
                transforms[name] = cpu_tensor(self.transforms[param])

        shared = {name: cpu_tensor(basis) for name, basis in self.shared_subspaces.items()}
        return {
            'eigens': eigens,
            'transforms': transforms,
            'shared_subspaces': shared,
            'latest_build_stats': self.latest_build_stats,
            'current_task_context': self.current_task_context,
        }

    def load_projection_state(self, projection_state, device):
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.shared_subspaces = {}
        for name, eigen in projection_state.get('eigens', {}).items():
            param = self.name_to_param.get(name)
            if param is None:
                continue
            self.eigens[param] = {
                'eigen_value': eigen['eigen_value'].to(device),
                'eigen_vector': eigen['eigen_vector'].to(device),
            }
        for name, transform in projection_state.get('transforms', {}).items():
            param = self.name_to_param.get(name)
            if param is None:
                continue
            self.transforms[param] = transform.to(device)
        for name, basis in projection_state.get('shared_subspaces', {}).items():
            self.shared_subspaces[name] = basis.to(device) if basis is not None else None
        self.latest_build_stats = projection_state.get('latest_build_stats', {})
        context = projection_state.get('current_task_context')
        if context is not None:
            self.set_task_context(
                task_index=context.get('task_index', 1),
                total_tasks=context.get('total_tasks'),
                alpha_t=context.get('alpha_t', 1.0),
                rho_t=context.get('rho_t', 1.0),
            )
