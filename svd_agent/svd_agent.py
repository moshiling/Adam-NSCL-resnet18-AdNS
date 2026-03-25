import re
from collections import defaultdict

import torch

import optim
from .agent import Agent


class SVDAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)
        self.drop_num = 0
        self.reg_params = {n: p for n, p in self.model.named_parameters() if 'bn' in n}
        self.empFI = False
        self.svd_lr = self.config['model_lr']
        self.init_model_optimizer()

    def init_model_optimizer(self):
        fea_params = [p for n, p in self.model.named_parameters() if not bool(re.match('last', n)) and 'bn' not in n]
        cls_params_all = list(p for n, p in self.model.named_children() if bool(re.match('last', n)))[0]
        cls_params = list(cls_params_all[str(self.task_count + 1)].parameters())
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
        model_optimizer_arg = {
            'params': [
                {
                    'params': fea_params,
                    'svd': True,
                    'lr': self.svd_lr,
                    'thres': self.config['svd_thres'],
                    'projection_mode': self.config.get('projection_mode', 'nscl'),
                    'sfcl_tau1': self.config.get('sfcl_tau1', 10.0),
                    'sfcl_tau2': self.config.get('sfcl_tau2', 10.0),
                    'sfcl_norm_projection': self.config.get('sfcl_norm_projection', True),
                    'use_shared_lowrank': self.config.get('use_shared_lowrank', False),
                    'shared_rank_mode': self.config.get('shared_rank_mode', 'avg'),
                    'shared_rank_ratio': self.config.get('shared_rank_ratio', 0.9),
                    'shared_subspace_mode': self.config.get('shared_subspace_mode', 'overlap_core'),
                    'shared_overlap_threshold': self.config.get('shared_overlap_threshold', 0.5),
                    'safe_boost': self.config.get('safe_boost', 1.25),
                    'risk_shrink': self.config.get('risk_shrink', 0.75),
                    'use_task_strength': self.config.get('use_task_strength', False),
                    'use_rho_t': self.config.get('use_rho_t', False),
                },
                {
                    'params': cls_params,
                    'weight_decay': 0.0,
                    'lr': self.config['head_lr'],
                },
                {
                    'params': bn_params,
                    'lr': self.config['bn_lr'],
                },
            ],
            'lr': self.config['model_lr'],
            'weight_decay': self.config['model_weight_decay'],
        }
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.model_optimizer,
            milestones=self.config['schedule'],
            gamma=self.config['gamma'],
        )
        if hasattr(self.model_optimizer, 'set_param_metadata'):
            self.model_optimizer.set_param_metadata({p: n for n, p in self.model.named_parameters()})

    def train_task(self, train_loader, val_loader=None, task_name=None):
        alpha_t, rho_t = self.prepare_optimizer_for_current_task()
        self.teacher_model, teacher_warmup = self.build_teacher_model(train_loader, task_name)
        epoch_metrics = self.train_model(
            train_loader,
            val_loader=val_loader,
            task_name=task_name,
            teacher_model=self.teacher_model,
        )
        self.latest_train_metrics = {
            'alpha_t': alpha_t,
            'rho_t': rho_t,
            'teacher_warmup': teacher_warmup,
            'epoch_metrics': epoch_metrics,
            'optimizer_task_stats': self.model_optimizer.get_task_stats() if hasattr(self.model_optimizer, 'get_task_stats') else {},
        }

        self.task_count += 1
        if self.task_count < self.num_task or self.num_task is None:
            if self.reset_model_optimizer:
                self.log('Classifier Optimizer is reset!')
                self.svd_lr = self.config['svd_lr']
                self.init_model_optimizer()
                self.model.zero_grad()
            with torch.no_grad():
                self.update_optim_transforms(train_loader)

            if self.reg_params:
                if len(self.regularization_terms) == 0:
                    self.regularization_terms = {'importance': defaultdict(list), 'task_param': defaultdict(list)}
                importance = self.calculate_importance(train_loader)
                for n, p in self.reg_params.items():
                    self.regularization_terms['importance'][n].append(importance[n].unsqueeze(0))
                    self.regularization_terms['task_param'][n].append(p.unsqueeze(0).clone().detach())
        self.teacher_model = None
        return self.latest_train_metrics

    def update_optim_transforms(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(m, 'weight') and not bool(re.match('last', n))]
        handles = []
        for module in modules:
            handles.append(module.register_forward_hook(hook=self.compute_cov))

        self.model.eval()
        for inputs, _, _ in train_loader:
            if self.config['gpu']:
                inputs = inputs.cuda(non_blocking=True)
            self.model.forward(inputs)
        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        build_stats = getattr(self.model_optimizer, 'latest_build_stats', {})
        for layer_name, stats in build_stats.items():
            singular = stats.get('singular', {})
            scale = stats.get('scale', {})
            safe_score = stats.get('safe_score', {})
            boost = stats.get('boost', {})
            self.log('Projector {} singular[min={:.6f}, max={:.6f}, mean={:.6f}] scale[min={:.6f}, max={:.6f}, mean={:.6f}] safe[min={:.6f}, max={:.6f}, mean={:.6f}] boost[min={:.6f}, max={:.6f}, mean={:.6f}] norm={:.6f}'.format(
                layer_name,
                singular.get('min', 0.0),
                singular.get('max', 0.0),
                singular.get('mean', 0.0),
                scale.get('min', 0.0),
                scale.get('max', 0.0),
                scale.get('mean', 0.0),
                safe_score.get('min', 0.0),
                safe_score.get('max', 0.0),
                safe_score.get('mean', 0.0),
                boost.get('min', 0.0),
                boost.get('max', 0.0),
                boost.get('mean', 0.0),
                stats.get('projector_norm', 0.0),
            ))

    def calculate_importance(self, dataloader):
        self.log('computing EWC')
        importance = {}
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)

        self.model.eval()
        for _, (inputs, targets, task) in enumerate(dataloader):
            if self.config['gpu']:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            output = self.model.forward(inputs)
            if self.empFI:
                ind = targets
            else:
                task_name = task[0] if self.multihead else 'ALL'
                pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:, :self.valid_out_dim]
                ind = pred.max(1)[1].flatten()

            loss = self.criterion(output, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.reg_params[n].grad is not None:
                    p += ((self.reg_params[n].grad ** 2) * len(inputs) / len(dataloader))
        return importance

    def reg_loss(self):
        self.reg_step += 1
        reg_loss = 0
        for n, p in self.reg_params.items():
            importance = torch.cat(self.regularization_terms['importance'][n], dim=0)
            old_params = torch.cat(self.regularization_terms['task_param'][n], dim=0)
            new_params = p.unsqueeze(0).expand(old_params.shape)
            reg_loss += (importance * (new_params - old_params) ** 2).sum()

        self.summarywritter.add_scalar('reg_loss', reg_loss, self.reg_step)
        return reg_loss

    def serialize_state(self):
        state = super().serialize_state()
        param_map = {p: n for n, p in self.model.named_parameters()}
        state['feature_covariance'] = {
            param_map[p]: cov.detach().cpu()
            for p, cov in self.fea_in.items()
            if torch.is_tensor(cov)
        }
        return state

    def load_serialized_state(self, state):
        super().load_serialized_state(state)
        name_to_param = {n: p for n, p in self.model.named_parameters()}
        self.fea_in = defaultdict(dict)
        for name, cov in state.get('feature_covariance', {}).items():
            if name in name_to_param:
                self.fea_in[name_to_param[name]] = cov.to(self.device)
