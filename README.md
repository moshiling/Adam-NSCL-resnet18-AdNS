# Adam-NSCL-resnet18-AdNS

PyTorch codebase for a continual-learning study built on top of Adam-NSCL, with a ResNet18/CIFAR100-10 setting and a soft-projection port toward SFCL-style optimization.

This repository contains three method lines:

- `nscl`: original Adam-NSCL hard projection baseline
- `sfcl`: full-space soft projection baseline implemented on top of the Adam-NSCL codebase
- `sfcl_adns`: SFCL-style soft projector plus AdNS-inspired modules

## What Was Changed

Relative to the original Adam-NSCL repository and paper, this repo adds:

- `projection_mode=sfcl`
  - uses all eigenvectors to build a full-space soft projector
  - does not fall back to candidate-nullspace-only hard projection
- `projection_mode=sfcl_adns`
  - keeps the SFCL-style `P = U diag(scale) U^T` backbone
  - adds shared-subspace modulation, task-aware projection strength, and intra-task distillation
- `shared_subspace_mode=union_lowrank | overlap_core`
  - both are implemented and checkpoint/resume compatible
- richer experiment outputs
  - `config_resolved.json`
  - `summary.json`
  - `task_metrics.jsonl`
  - `plasticity_stats.jsonl`
  - `train_loss_breakdown.jsonl`

Important scope note:

- This repo does **not** implement the SSL / self-supervision orthogonal enhancement branch sometimes associated with SFCL-style methods.
- The implemented method is best described as:
  - an SFCL-style full-space soft projection framework with shared-subspace modulation, task-strength scheduling, and intra-task distillation

## Main Setting

- Dataset: `CIFAR100`
- Split: `10/10`
- Backbone: `ResNet18`
- Optimizer family: `Adam`
- Training schedule: `30 60 80`
- Baseline threshold: `svd_thres = 10.0`

The historical baseline log from the original project setup recovered:

- ACC: `73.37`
- BWT: `-1.6444`

This historical result is provided as reference only. It is not perfectly apples-to-apples with the corrected runs in this repository because the scheduler stepping logic was fixed during this port.

## Most Trustworthy Formal Results

These are the corrected full runs used as the main comparison table.

| Method | Experiment | ACC | BWT |
| --- | --- | ---: | ---: |
| `nscl` | `nscl_full80_seed0_fixed_v2` | 72.95 | -1.80 |
| `sfcl` | `sfcl_full80_seed0_fixed` | 74.25 | -4.4778 |
| `sfcl_adns` + `overlap_core` | `sfcl_adns_full80_seed0_fixed_v3_gpu1` | 49.51 | -40.90 |
| `sfcl_adns` + `union_lowrank` | `sfcl_adns_union_lowrank_full80_seed0` | 52.44 | -37.5333 |

Takeaway:

- `sfcl` gives the best ACC in the current single-seed main table.
- `nscl` remains the most stable result in terms of BWT.
- the current `sfcl_adns` full combination is clearly worse than both `nscl` and `sfcl`.
- among shared-subspace variants, `union_lowrank` is better than `overlap_core` in this setting.

## Union-Lowrank Module Ablation

To locate what hurt `sfcl_adns`, the shared-subspace mode was fixed to `union_lowrank` and modules were disentangled.

### Seed 0

| ID | Configuration | ACC | BWT |
| --- | --- | ---: | ---: |
| E1 | shared only | 74.36 | -3.5111 |
| E2 | shared + task_strength | 38.88 | -53.8889 |
| E3 | shared + KD | 74.00 | -2.2333 |
| E4 | shared + task_strength + KD | 52.44 | -37.5333 |

### Seed 1

| ID | Configuration | ACC | BWT |
| --- | --- | ---: | ---: |
| E1 | shared only | 74.17 | -3.4111 |
| E3 | shared + KD | 73.63 | -2.1889 |

Main conclusion from the ablation:

- `union_lowrank` shared-subspace by itself is not the main problem
- KD is not the main damage source and may mildly help stability
- `task_strength` is the dominant degradation source in the current implementation

## Key Files

### Core code

- [main.py](main.py)
- [optim/adam_svd.py](optim/adam_svd.py)
- [optim/projection_builder.py](optim/projection_builder.py)
- [utils/subspace_utils.py](utils/subspace_utils.py)
- [utils/schedule_utils.py](utils/schedule_utils.py)
- [utils/distill_utils.py](utils/distill_utils.py)
- [svd_agent/agent.py](svd_agent/agent.py)
- [svd_agent/svd_agent.py](svd_agent/svd_agent.py)

### Configs

- [configs/sfcl_resnet18_cifar100_10.yaml](configs/sfcl_resnet18_cifar100_10.yaml)
- [configs/sfcl_adns_resnet18_cifar100_10.yaml](configs/sfcl_adns_resnet18_cifar100_10.yaml)

### Main reports

- [results/final_repro_round/final_report.md](results/final_repro_round/final_report.md)
- [results/final_repro_round/experiment_table.csv](results/final_repro_round/experiment_table.csv)
- [results/ablation_unionlowrank_round1/final_report.md](results/ablation_unionlowrank_round1/final_report.md)
- [results/ablation_unionlowrank_round1/status_readable.md](results/ablation_unionlowrank_round1/status_readable.md)
- [results/method_alignment_fix_round/final_report.md](results/method_alignment_fix_round/final_report.md)
- [results/method_alignment_audit/method_alignment_report.md](results/method_alignment_audit/method_alignment_report.md)

## Selected Result Files

### Main full runs

- [results/final_repro_round/nscl_full80_seed0_fixed_v2/summary.json](results/final_repro_round/nscl_full80_seed0_fixed_v2/summary.json)
- [results/final_repro_round/sfcl_full80_seed0_fixed/summary.json](results/final_repro_round/sfcl_full80_seed0_fixed/summary.json)
- [results/final_repro_round/sfcl_adns_full80_seed0_fixed_v3_gpu1/summary.json](results/final_repro_round/sfcl_adns_full80_seed0_fixed_v3_gpu1/summary.json)
- [results/final_repro_round/sfcl_adns_union_lowrank_full80_seed0/summary.json](results/final_repro_round/sfcl_adns_union_lowrank_full80_seed0/summary.json)

### Union-lowrank ablation

- [results/ablation_unionlowrank_round1/union_shared_only_full80_seed0/summary.json](results/ablation_unionlowrank_round1/union_shared_only_full80_seed0/summary.json)
- [results/ablation_unionlowrank_round1/union_shared_taskstrength_full80_seed0/summary.json](results/ablation_unionlowrank_round1/union_shared_taskstrength_full80_seed0/summary.json)
- [results/ablation_unionlowrank_round1/union_shared_kd_full80_seed0/summary.json](results/ablation_unionlowrank_round1/union_shared_kd_full80_seed0/summary.json)
- [results/ablation_unionlowrank_round1/union_shared_only_full80_seed1/summary.json](results/ablation_unionlowrank_round1/union_shared_only_full80_seed1/summary.json)
- [results/ablation_unionlowrank_round1/union_shared_kd_full80_seed1/summary.json](results/ablation_unionlowrank_round1/union_shared_kd_full80_seed1/summary.json)

## Quick Start

Original script:

```bash
sh scripts_svd/adamnscl.sh
```

Example config-based run:

```bash
python main.py \
  --config configs/sfcl_adns_resnet18_cifar100_10.yaml \
  --experiment_name demo_sfcl_adns \
  --output_root results/demo
```

## Original Adam-NSCL Citation

```bibtex
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Shipeng and Li, Xiaorong and Sun, Jian and Xu, Zongben},
    title     = {Training Networks in Null Space of Feature Covariance for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {184-193}
}
```
