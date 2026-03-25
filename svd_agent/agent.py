import time
from datetime import datetime
from types import MethodType
from collections import Counter

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import optim
from utils.distill_utils import compute_intra_task_distill_loss, warmup_teacher_head
from utils.metric import AverageMeter, Timer, accumulate_acc
from utils.schedule_utils import get_alpha_t, get_rho_t
from utils.utils import count_parameter, factory


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Counter):
        return Counter({k: _to_cpu(v) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Counter):
        return Counter({k: _move_to_device(v, device) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    return value


class Agent(nn.Module):
    def __init__(self, agent_config):
        super().__init__()
        self.log = print if agent_config['print_freq'] > 0 else (lambda *args: None)
        self.config = agent_config
        self.log(agent_config)

        self.multihead = True if len(self.config['out_dim']) > 1 else False
        self.num_task = len(self.config['out_dim']) if len(self.config['out_dim']) > 1 else None
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.valid_out_dim = 'ALL'
        self.clf_param_num = count_parameter(self.model)
        self.task_count = 0
        self.reg_step = 0
        tensorboard_dir = self.config.get('tensorboard_dir') or ('./log/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.summarywritter = SummaryWriter(tensorboard_dir)
        self.device = torch.device('cuda' if self.config['gpu'] else 'cpu')
        if self.config['gpu']:
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
        self.log('#param of model:{}'.format(self.clf_param_num))
        self.reset_model_optimizer = agent_config['reset_model_opt']
        self.dataset_name = agent_config['dataset_name']
        self.regularization_terms = {}
        self.teacher_model = None
        self.latest_train_metrics = {}
        self.latest_teacher_warmup = []

    def create_model(self):
        cfg = self.config
        model = factory('models', cfg['model_type'], cfg['model_name'])()
        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim, bias=True)

        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        model.logits = MethodType(new_logits, model)
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'], map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def init_model_optimizer(self):
        model_optimizer_arg = {
            'params': self.model.parameters(),
            'lr': self.config['model_lr'],
            'weight_decay': self.config['model_weight_decay'],
        }
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
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

    def _current_alpha_rho(self):
        task_index = self.task_count + 1
        total_tasks = self.num_task
        alpha_t = 1.0
        rho_t = 1.0
        if self.config.get('use_task_strength', False):
            alpha_t = get_alpha_t(
                task_index,
                total_tasks,
                alpha_min=self.config.get('alpha_min', 0.3),
                alpha_max=self.config.get('alpha_max', 0.9),
                schedule=self.config.get('alpha_schedule', 'linear'),
            )
        if self.config.get('use_rho_t', False):
            rho_t = get_rho_t(
                task_index,
                total_tasks,
                rho_min=self.config.get('rho_min', 1.0),
                rho_max=self.config.get('rho_max', 1.0),
                schedule=self.config.get('alpha_schedule', 'linear'),
            )
        return alpha_t, rho_t

    def prepare_optimizer_for_current_task(self):
        alpha_t, rho_t = self._current_alpha_rho()
        if hasattr(self.model_optimizer, 'set_task_context'):
            self.model_optimizer.set_task_context(
                task_index=self.task_count + 1,
                total_tasks=self.num_task,
                alpha_t=alpha_t,
                rho_t=rho_t,
            )
        if hasattr(self.model_optimizer, 'reset_task_stats'):
            self.model_optimizer.reset_task_stats()
        self.log('Task {} alpha_t {:.4f} rho_t {:.4f} distill {}'.format(
            self.task_count + 1, alpha_t, rho_t, self.config.get('use_intra_task_distill', False)))
        return alpha_t, rho_t

    def build_teacher_model(self, train_loader, task_name):
        if not self.config.get('use_intra_task_distill', False):
            self.latest_teacher_warmup = []
            return None, []
        if self.task_count <= 0:
            self.latest_teacher_warmup = []
            return None, []
        teacher, metrics = warmup_teacher_head(
            self.model,
            train_loader,
            task_name,
            device=self.device,
            epochs=self.config.get('teacher_warmup_epochs', 5),
            lr=self.config.get('head_lr', 1e-3),
            weight_decay=0.0,
            log_fn=self.log,
        )
        self.latest_teacher_warmup = metrics
        return teacher, metrics

    def train_task(self, train_loader, val_loader=None, task_name=None):
        raise NotImplementedError

    def train_epoch(self, train_loader, epoch, task_name, teacher_model=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ce_losses = AverageMeter()
        distill_losses = AverageMeter()
        acc = AverageMeter()
        end = time.time()
        for i, (inputs, target, task) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if self.config['gpu']:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = self.model.forward(inputs)
            ce_loss = self.criterion(output, target, task)
            distill_loss = torch.zeros(1, device=self.device).squeeze()
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)[task_name]
                distill_loss = compute_intra_task_distill_loss(
                    output[task_name],
                    teacher_logits,
                    tau_distill=self.config.get('tau_distill', 2.0),
                )
            loss = ce_loss + self.config.get('beta_distill', 0.5) * distill_loss

            acc = accumulate_acc(output, target, task, acc)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(float(loss.detach().cpu().item()), inputs.size(0))
            ce_losses.update(float(ce_loss.detach().cpu().item()), inputs.size(0))
            distill_losses.update(float(distill_loss.detach().cpu().item()), inputs.size(0))

            if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(train_loader):
                self.log(
                    '[{0}/{1}]\t{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                    '{loss.val:.3f} ({loss.avg:.3f})\t'
                    'CE {ce.val:.3f} ({ce.avg:.3f})\t'
                    'KD {kd.val:.3f} ({kd.avg:.3f})\t'
                    '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        ce=ce_losses,
                        kd=distill_losses,
                        acc=acc,
                    )
                )
        self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
        return {
            'loss': losses.avg,
            'acc': acc.avg,
            'ce_loss': ce_losses.avg,
            'distill_loss': distill_losses.avg,
            'total_loss': losses.avg,
        }

    def train_model(self, train_loader, val_loader=None, task_name=None, teacher_model=None):
        epoch_metrics = []
        for epoch in range(self.config['schedule'][-1]):
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.model_optimizer.param_groups:
                self.log('LR:', param_group['lr'])
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            metrics = self.train_epoch(train_loader, epoch, task_name=task_name, teacher_model=teacher_model)
            metrics['epoch'] = epoch
            if val_loader is not None:
                metrics['val_acc'] = self.validation(val_loader)
            epoch_metrics.append(metrics)
            self.model_scheduler.step()
        return epoch_metrics

    def validation(self, dataloader):
        batch_timer = Timer()
        val_acc = AverageMeter()
        losses = AverageMeter()
        batch_timer.tic()
        self.model.eval()
        for _, (inputs, target, task) in enumerate(dataloader):
            with torch.no_grad():
                if self.config['gpu']:
                    inputs = inputs.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                output = self.model.forward(inputs)
                loss = self.criterion(output, target, task, regularization=False)
            losses.update(float(loss.detach().cpu().item()), inputs.size(0))
            for t in output.keys():
                output[t] = output[t].detach()
            val_acc = accumulate_acc(output, target, task, val_acc)
        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'.format(acc=val_acc, time=batch_timer.toc()))
        self.log(' * Val loss {loss.avg:.3f}, Total time {time:.2f}'.format(loss=losses, time=batch_timer.toc()))
        return val_acc.avg

    def criterion(self, preds, targets, tasks, regularization=True):
        loss = self.cross_entropy(preds, targets, tasks)
        if regularization and len(self.regularization_terms) > 0:
            reg_loss = self.reg_loss()
            loss += self.config['reg_coef'] * reg_loss
        return loss

    def cross_entropy(self, preds, targets, tasks):
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)
            loss /= len(targets)
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim, int):
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def add_valid_output_dim(self, dim=0):
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def serialize_state(self):
        return {
            'model': _to_cpu(self.model.state_dict()),
            'optimizer': _to_cpu(self.model_optimizer.state_dict()),
            'scheduler': _to_cpu(self.model_scheduler.state_dict()),
            'task_count': self.task_count,
            'valid_out_dim': self.valid_out_dim,
            'regularization_terms': _to_cpu(self.regularization_terms),
            'reg_step': self.reg_step,
            'projection_state': self.model_optimizer.serialize_projection_state() if hasattr(self.model_optimizer, 'serialize_projection_state') else {},
        }

    def load_serialized_state(self, state):
        self.model.load_state_dict(state['model'])
        self.task_count = state.get('task_count', 0)
        self.valid_out_dim = state.get('valid_out_dim', 'ALL')
        self.regularization_terms = _move_to_device(state.get('regularization_terms', {}), self.device)
        self.reg_step = state.get('reg_step', 0)
        self.init_model_optimizer()
        self.model_optimizer.load_state_dict(state['optimizer'])
        for optimizer_state in self.model_optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        scheduler_state = state['scheduler']
        milestones = scheduler_state.get('milestones')
        if isinstance(milestones, dict) and not isinstance(milestones, Counter):
            scheduler_state = dict(scheduler_state)
            scheduler_state['milestones'] = Counter(milestones)
        self.model_scheduler.load_state_dict(scheduler_state)
        if hasattr(self.model_optimizer, 'load_projection_state'):
            self.model_optimizer.load_projection_state(state.get('projection_state', {}), self.device)
