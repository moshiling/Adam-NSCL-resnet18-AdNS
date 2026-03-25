import argparse
import json
import os
import random
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import yaml

from dataloaders.datasetGen import PermutedGen, SplitGen
from utils.utils import factory


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_serializable(value):
    if isinstance(value, OrderedDict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def append_jsonl(path, payload):
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(to_serializable(payload), ensure_ascii=False) + '\n')


def load_yaml_config(path):
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def infer_run_name(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return '{}_{}_{}_seed{}'.format(args.projection_mode, args.model_name, args.dataset, args.seed) + '_' + timestamp


def auto_select_gpu(min_free_gb=4.0):
    try:
        command = [
            'nvidia-smi',
            '--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits',
        ]
        result = subprocess.check_output(command, text=True).strip().splitlines()
        candidates = []
        for line in result:
            index, free_mem, util = [part.strip() for part in line.split(',')]
            candidates.append((int(index), float(free_mem), float(util)))
        if not candidates:
            return None
        eligible = [item for item in candidates if item[1] >= min_free_gb * 1024]
        if eligible:
            eligible.sort(key=lambda item: (item[2], -item[1], item[0]))
            return eligible[0][0]
        candidates.sort(key=lambda item: (item[2], -item[1], item[0]))
        return candidates[0][0]
    except Exception:
        return None


def build_parser(defaults=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='results/adns_sfcl_port/runs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--auto_select_gpu', dest='auto_select_gpu', action='store_true')
    parser.add_argument('--no_auto_select_gpu', dest='auto_select_gpu', action='store_false')
    parser.set_defaults(auto_select_gpu=True)
    parser.add_argument('--min_gpu_free_gb', type=float, default=4.0)

    parser.add_argument('--gpuid', nargs='+', type=int, default=[0], help='The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only')
    parser.add_argument('--model_type', type=str, default='resnet', help='The type (mlp|lenet|vgg|resnet) of backbone network')
    parser.add_argument('--model_name', type=str, default='resnet18', help='The name of actual model for the backbone')
    parser.add_argument('--force_out_dim', type=int, default=0, help='Set 0 to let the task decide the required output dimension')
    parser.add_argument('--agent_type', type=str, default='svd_based', help='The type (filename) of agent')
    parser.add_argument('--agent_name', type=str, default='svd_based', help='The class name of agent')
    parser.add_argument('--model_optimizer', type=str, default='Adam', help='SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...')
    parser.add_argument('--dataroot', type=str, default='../data', help='The root folder of dataset or downloaded data')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='MNIST(default)|CIFAR10|CIFAR100')
    parser.add_argument('--n_permutation', type=int, default=0, help='Enable permuted tests when >0')
    parser.add_argument('--first_split_size', type=int, default=10)
    parser.add_argument('--other_split_size', type=int, default=10)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true')
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_false')
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0, help='#Thread for dataloader')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_lr', type=float, default=0.0005, help='Classifier Learning rate')
    parser.add_argument('--head_lr', type=float, default=0.0005, help='Classifier Learning rate')
    parser.add_argument('--svd_lr', type=float, default=0.0005, help='Classifier Learning rate')
    parser.add_argument('--bn_lr', type=float, default=0.0005, help='Classifier Learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay')
    parser.add_argument('--svd_thres', type=float, default=1.0, help='reserve eigenvector')
    parser.add_argument('--a', type=float, default=None, help='Alias of svd_thres for compatibility')
    parser.add_argument('--thres', type=float, default=None, help='Alias of svd_thres for compatibility')
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--model_weight_decay', type=float, default=1e-5)
    parser.add_argument('--schedule', nargs='+', type=int, default=[1], help='epoch ')
    parser.add_argument('--print_freq', type=float, default=10, help='Print the log at every x iteration')
    parser.add_argument('--model_weights', type=str, default=None, help='The path to the file for the model weights (*.pth).')
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true')
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat the experiment N times')
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true')
    parser.add_argument('--with_head', dest='with_head', default=False, action='store_true')
    parser.add_argument('--reset_model_opt', dest='reset_model_opt', default=True, action='store_true')
    parser.add_argument('--reg_coef', type=float, default=100)

    parser.add_argument('--projection_mode', type=str, default='nscl', choices=['nscl', 'sfcl', 'sfcl_adns'])
    parser.add_argument('--sfcl_tau1', type=float, default=10.0)
    parser.add_argument('--sfcl_tau2', type=float, default=10.0)
    parser.add_argument('--sfcl_norm_projection', dest='sfcl_norm_projection', action='store_true')
    parser.add_argument('--no_sfcl_norm_projection', dest='sfcl_norm_projection', action='store_false')
    parser.set_defaults(sfcl_norm_projection=True)
    parser.add_argument('--use_shared_lowrank', dest='use_shared_lowrank', action='store_true')
    parser.add_argument('--no_shared_lowrank', dest='use_shared_lowrank', action='store_false')
    parser.set_defaults(use_shared_lowrank=True)
    parser.add_argument('--shared_subspace_mode', type=str, default='overlap_core', choices=['union_lowrank', 'overlap_core'])
    parser.add_argument('--shared_rank_mode', type=str, default='avg', choices=['max', 'avg', 'min'])
    parser.add_argument('--shared_rank_ratio', type=float, default=0.9)
    parser.add_argument('--shared_overlap_threshold', type=float, default=0.5)
    parser.add_argument('--safe_boost', type=float, default=1.25)
    parser.add_argument('--risk_shrink', type=float, default=0.75)
    parser.add_argument('--lambda_share', type=float, default=0.5)
    parser.add_argument('--use_task_strength', dest='use_task_strength', action='store_true')
    parser.add_argument('--no_task_strength', dest='use_task_strength', action='store_false')
    parser.set_defaults(use_task_strength=True)
    parser.add_argument('--alpha_min', type=float, default=0.3)
    parser.add_argument('--alpha_max', type=float, default=0.9)
    parser.add_argument('--alpha_schedule', type=str, default='linear', choices=['linear', 'cosine', 'exp'])
    parser.add_argument('--use_rho_t', dest='use_rho_t', action='store_true')
    parser.add_argument('--no_rho_t', dest='use_rho_t', action='store_false')
    parser.set_defaults(use_rho_t=False)
    parser.add_argument('--rho_min', type=float, default=1.0)
    parser.add_argument('--rho_max', type=float, default=1.0)
    parser.add_argument('--use_intra_task_distill', dest='use_intra_task_distill', action='store_true')
    parser.add_argument('--no_intra_task_distill', dest='use_intra_task_distill', action='store_false')
    parser.set_defaults(use_intra_task_distill=False)
    parser.add_argument('--teacher_warmup_epochs', type=int, default=5)
    parser.add_argument('--beta_distill', type=float, default=0.5)
    parser.add_argument('--tau_distill', type=float, default=2.0)
    parser.set_defaults(**(defaults or {}))
    return parser


def get_args(argv):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = load_yaml_config(config_args.config)
    parser = build_parser(defaults=defaults)
    args = parser.parse_args(argv)
    if args.a is not None:
        args.svd_thres = args.a
    if args.thres is not None:
        args.svd_thres = args.thres
    return args


def set_random_seed(seed, use_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_datasets(args):
    train_dataset, val_dataset = factory('dataloaders', 'base', args.dataset)(args.dataroot, args.train_aug)
    if args.n_permutation > 0:
        return PermutedGen(
            train_dataset,
            val_dataset,
            args.n_permutation,
            remap_class=not args.no_class_remap,
        )
    return SplitGen(
        train_dataset,
        val_dataset,
        first_split_sz=args.first_split_size,
        other_split_sz=args.other_split_size,
        rand_split=args.rand_split,
        remap_class=not args.no_class_remap,
    )


def create_agent(args, task_output_space, run_dir):
    dataset_name = args.dataset + '_{}_{}'.format(args.first_split_size, args.other_split_size)
    agent_config = {
        'model_lr': args.model_lr,
        'momentum': args.momentum,
        'model_weight_decay': args.model_weight_decay,
        'schedule': args.schedule,
        'model_type': args.model_type,
        'model_name': args.model_name,
        'model_weights': args.model_weights,
        'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
        'model_optimizer': args.model_optimizer,
        'print_freq': args.print_freq,
        'gpu': True if args.gpuid[0] >= 0 else False,
        'with_head': args.with_head,
        'reset_model_opt': args.reset_model_opt,
        'reg_coef': args.reg_coef,
        'head_lr': args.head_lr,
        'svd_lr': args.svd_lr,
        'bn_lr': args.bn_lr,
        'svd_thres': args.svd_thres,
        'gamma': args.gamma,
        'dataset_name': dataset_name,
        'projection_mode': args.projection_mode,
        'sfcl_tau1': args.sfcl_tau1,
        'sfcl_tau2': args.sfcl_tau2,
        'sfcl_norm_projection': args.sfcl_norm_projection,
        'use_shared_lowrank': args.use_shared_lowrank and args.projection_mode == 'sfcl_adns',
        'shared_rank_mode': args.shared_rank_mode,
        'shared_rank_ratio': args.shared_rank_ratio,
        'shared_subspace_mode': args.shared_subspace_mode,
        'shared_overlap_threshold': args.shared_overlap_threshold,
        'safe_boost': args.safe_boost,
        'risk_shrink': args.risk_shrink,
        'use_task_strength': args.use_task_strength and args.projection_mode == 'sfcl_adns',
        'alpha_min': args.alpha_min,
        'alpha_max': args.alpha_max,
        'alpha_schedule': args.alpha_schedule,
        'use_rho_t': args.use_rho_t and args.projection_mode == 'sfcl_adns',
        'rho_min': args.rho_min,
        'rho_max': args.rho_max,
        'use_intra_task_distill': args.use_intra_task_distill and args.projection_mode == 'sfcl_adns',
        'teacher_warmup_epochs': args.teacher_warmup_epochs,
        'beta_distill': args.beta_distill,
        'tau_distill': args.tau_distill,
        'tensorboard_dir': ensure_dir(os.path.join(run_dir, 'tensorboard')),
    }
    agent = factory('svd_agent', args.agent_type, args.agent_name)(agent_config)
    return agent


def compute_histories(acc_table, task_names):
    avg_acc_history = [0] * len(task_names)
    bwt_history = [0] * len(task_names)
    for i in range(len(task_names)):
        train_name = task_names[i]
        cls_acc_sum = 0
        backward_transfer = 0
        for j in range(i + 1):
            val_name = task_names[j]
            cls_acc_sum += acc_table[val_name][train_name]
            backward_transfer += acc_table[val_name][train_name] - acc_table[val_name][val_name]
        avg_acc_history[i] = cls_acc_sum / (i + 1)
        bwt_history[i] = backward_transfer / i if i > 0 else 0
    return avg_acc_history, bwt_history


def save_checkpoint(path, agent, acc_table, task_names, next_task_idx, task_metrics_history, plasticity_history, args):
    checkpoint = {
        'agent_state': agent.serialize_state(),
        'acc_table': to_serializable(acc_table),
        'task_names': task_names,
        'next_task_idx': next_task_idx,
        'task_metrics_history': task_metrics_history,
        'plasticity_history': plasticity_history,
        'args': vars(args),
    }
    torch.save(checkpoint, path)


def run(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    should_auto_select = args.auto_select_gpu and torch.cuda.is_available() and (len(args.gpuid) == 0 or args.gpuid[0] < 0)
    if should_auto_select:
        selected = auto_select_gpu(args.min_gpu_free_gb)
        if selected is not None:
            args.gpuid = [selected]

    if torch.cuda.is_available() and args.gpuid[0] >= 0:
        torch.cuda.set_device(args.gpuid[0])
        use_cuda = True
    else:
        args.gpuid = [-1]
        use_cuda = False
    set_random_seed(args.seed, use_cuda)

    run_name = args.experiment_name or infer_run_name(args)
    run_dir = ensure_dir(os.path.join(project_root, args.output_root, run_name))
    ensure_dir(os.path.join(run_dir, 'checkpoints'))
    save_json(os.path.join(run_dir, 'config_resolved.json'), vars(args))

    train_dataset_splits, val_dataset_splits, task_output_space = prepare_datasets(args)
    agent = create_agent(args, task_output_space, run_dir)
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)

    acc_table = OrderedDict()
    task_metrics_history = []
    plasticity_history = []
    start_task_idx = 0

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        agent.load_serialized_state(checkpoint['agent_state'])
        task_names = checkpoint.get('task_names', task_names)
        loaded_acc_table = checkpoint.get('acc_table', {})
        for outer_key in task_names:
            if outer_key in loaded_acc_table:
                acc_table[outer_key] = OrderedDict(loaded_acc_table[outer_key])
        task_metrics_history = checkpoint.get('task_metrics_history', [])
        plasticity_history = checkpoint.get('plasticity_history', [])
        start_task_idx = checkpoint.get('next_task_idx', 0)

    for i, train_name in enumerate(task_names):
        if i < start_task_idx:
            continue
        print('======================', train_name, '=======================')
        train_loader = torch.utils.data.DataLoader(
            train_dataset_splits[train_name],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_splits[train_name],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        if args.incremental_class:
            agent.add_valid_output_dim(task_output_space[train_name])

        train_metrics = agent.train_task(train_loader, val_loader, task_name=train_name)
        torch.cuda.empty_cache()
        acc_table[train_name] = OrderedDict()
        for j in range(i + 1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
            val_loader_eval = torch.utils.data.DataLoader(
                val_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            acc_table[val_name][train_name] = agent.validation(val_loader_eval)
            print('**************************************************')

        avg_acc_history, bwt_history = compute_histories(acc_table, task_names[:i + 1])
        task_summary = {
            'task': train_name,
            'task_index': i + 1,
            'avg_acc': avg_acc_history[-1],
            'bwt': bwt_history[-1],
            'acc_by_eval_task': {name: acc_table[name][train_name] for name in task_names[:i + 1]},
            'train_metrics': train_metrics,
            'projection_build_stats': getattr(agent.model_optimizer, 'latest_build_stats', {}),
        }
        task_metrics_history.append(task_summary)
        plasticity_payload = {
            'task': train_name,
            'task_index': i + 1,
            'projection_mode': args.projection_mode,
            'alpha_t': train_metrics.get('alpha_t', 1.0),
            'rho_t': train_metrics.get('rho_t', 1.0),
            'optimizer_task_stats': train_metrics.get('optimizer_task_stats', {}),
            'projection_build_stats': getattr(agent.model_optimizer, 'latest_build_stats', {}),
        }
        plasticity_history.append(plasticity_payload)
        append_jsonl(os.path.join(run_dir, 'task_metrics.jsonl'), task_summary)
        append_jsonl(os.path.join(run_dir, 'plasticity_stats.jsonl'), plasticity_payload)
        teacher_warmup = train_metrics.get('teacher_warmup', [])
        if teacher_warmup:
            save_json(
                os.path.join(run_dir, 'teacher_warmup_metrics_task_{:02d}.json'.format(i + 1)),
                {
                    'task': train_name,
                    'task_index': i + 1,
                    'task_name': train_name,
                    'teacher_warmup_metrics': teacher_warmup,
                },
            )
        for epoch_payload in train_metrics.get('epoch_metrics', []):
            append_jsonl(
                os.path.join(run_dir, 'train_loss_breakdown.jsonl'),
                {
                    'task': train_name,
                    'task_index': i + 1,
                    'epoch': epoch_payload.get('epoch'),
                    'acc': epoch_payload.get('acc'),
                    'ce_loss': epoch_payload.get('ce_loss'),
                    'distill_loss': epoch_payload.get('distill_loss'),
                    'total_loss': epoch_payload.get('total_loss', epoch_payload.get('loss')),
                    'loss': epoch_payload.get('loss'),
                    'val_acc': epoch_payload.get('val_acc'),
                },
            )

        save_checkpoint(
            os.path.join(run_dir, 'checkpoints', 'last.pt'),
            agent,
            acc_table,
            task_names,
            i + 1,
            task_metrics_history,
            plasticity_history,
            args,
        )
        save_checkpoint(
            os.path.join(run_dir, 'checkpoints', 'task_{:02d}.pt'.format(i + 1)),
            agent,
            acc_table,
            task_names,
            i + 1,
            task_metrics_history,
            plasticity_history,
            args,
        )

    avg_acc_history, bwt_history = compute_histories(acc_table, task_names)
    summary = {
        'run_name': run_name,
        'run_dir': run_dir,
        'projection_mode': args.projection_mode,
        'task_order': task_names,
        'acc_table': acc_table,
        'avg_acc_history': avg_acc_history,
        'bwt_history': bwt_history,
        'final_avg_acc': avg_acc_history[-1],
        'final_bwt': bwt_history[-1],
        'seed': args.seed,
        'gpuid': args.gpuid,
    }
    save_json(os.path.join(run_dir, 'summary.json'), summary)

    print(acc_table)
    for idx, task_name in enumerate(task_names):
        print('Task', task_name, 'average acc:', avg_acc_history[idx])
        print('Task', task_name, 'backward transfer:', bwt_history[idx])
    print('===Summary of experiment repeats: 1 / 1 ===')
    print('The last avg acc of all repeats:', np.array([avg_acc_history[-1]]))
    print('The last bwt of all repeats:', np.array([bwt_history[-1]]))
    print('acc mean:', avg_acc_history[-1], 'acc std:', 0.0)
    print('bwt mean:', bwt_history[-1], 'bwt std:', 0.0)
    return summary


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)
