# Adam-SFCL / sfcl_adns 最终报告

## 1. 最终采用的项目根目录

- 实际项目根目录：
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`
- baseline 日志：
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /train_thres_10.log`

## 2. baseline 日志中恢复出的关键设定

- 数据集：
  - `dataset=CIFAR100`
  - `first_split_size=10`
  - `other_split_size=10`
- backbone：
  - `model_type=resnet`
  - `model_name=resnet18`
- 优化器与学习率：
  - `model_optimizer=Adam`
  - `model_lr=1e-4`
  - `head_lr=1e-3`
  - `svd_lr=5e-5`
  - `bn_lr=5e-4`
  - `model_weight_decay=5e-5`
- 训练长度与 schedule：
  - `schedule=[30,60,80]`
  - `gamma=0.5`
  - `batch_size=32`
- approximate null space 阈值：
  - `svd_thres=10`
- 历史 baseline 最终结果：
  - `ACC=73.37`
  - `BWT=-1.644444444444441`

## 3. 从 Adam-NSCL 到 Adam-SFCL 的映射

### 3.1 原始 Adam-NSCL

- 用累计的 uncentered feature covariance 做 SVD：
  - `C = U diag(sigma) U^T`
- 通过阈值规则选出小奇异值方向：
  - `U2 = {u_i | sigma_i <= sigma_min * thres}`
- 构造硬投影：
  - `P_nscl = U2 U2^T`
- 在当前工程中，真正被右乘投影的是 Adam update，而不是裸梯度：
  - `g_proj = (P_nscl / ||P_nscl||) g_adam`

### 3.2 新增 Adam-SFCL baseline

- 仍然复用同一份 covariance、同一份 SVD、同一份阈值切分
- 不再只保留 `U2`，而是使用全空间 `U=[U1,U2]`
- 构造软投影：
  - `P_sfcl = U diag(scale_all) U^T`
- 当前实现采用两段式缩放：
  - `scale(U1)=1/(1+tau1)`
  - `scale(U2)=tau2/(1+tau2)`
- 再按配置决定是否归一化：
  - `g_sfcl = (P_sfcl / ||P_sfcl||) g_adam`

## 4. 从 AdNS 到 sfcl_adns 的迁移

### 4.1 模块 A：Shared Low-Rank Null Space -> Shared Safe Subspace

- 原始思路：
  - `U_cat=[U_pre, U_cur]`
  - 低秩近似得到共享子空间
- 迁移后的 soft-friendly 版本：
  - 先保留 `P_sfcl` 的全空间软投影基底
  - 再利用 `U_share` 与各奇异向量的 overlap，得到每个奇异方向的 `safe_score`
  - 最终缩放：
    - `final_scale = base_sfcl_scale * boost(safe_score)`
  - 其中：
    - safe 方向更大
    - risk 方向更小

### 4.2 模块 B：Non-uniform Constraint Strength -> Task-aware Soft Projection Strength

- 原始 AdNS 是任务数递增时收紧硬约束
- 当前迁移成软投影混合强度：
  - `g_final = (1-alpha_t) * g_raw + alpha_t * g_sfcl`
- 支持：
  - `linear`
  - `cosine`
  - `exp`

### 4.3 模块 C：Intra-task Distillation

- 对 `t>1` 的任务：
  - 复制 teacher model
  - 冻结 backbone，仅 warmup 当前 task head
  - 正式训练时加入：
    - `L = L_ce + beta_distill * L_kl`
- 当前工程优先落在多头路径上：
  - 只蒸馏当前任务 head 的 logits

## 5. 关键代码修改点

- `main.py`
  - 新增 yaml config、checkpoint/resume、summary/jsonl、run dir 管理
- `optim/adam_svd.py`
  - 新增 `projection_mode=nscl|sfcl|sfcl_adns`
  - 新增 shared subspace / task strength / projector state serialize
- `optim/projection_builder.py`
  - 新增三个 projector builder
- `utils/subspace_utils.py`
  - 新增 candidate nullspace / shared low-rank / angle 工具
- `utils/schedule_utils.py`
  - 新增 `get_alpha_t` / `get_rho_t`
- `utils/distill_utils.py`
  - 新增 teacher warmup 与 KL distillation
- `svd_agent/agent.py`
  - 新增 distillation 训练逻辑与 agent state 恢复
- `svd_agent/svd_agent.py`
  - 新增 projection 配置注入、feature covariance checkpoint、projector stats

## 6. 关键超参数

- baseline 阈值：
  - `svd_thres=10`
- SFCL：
  - `sfcl_tau1=10`
  - `sfcl_tau2=10`
  - `sfcl_norm_projection=true`
- shared low-rank：
  - `shared_rank_mode=avg`
  - `shared_rank_ratio=0.9`
  - `safe_boost=1.25`
  - `risk_shrink=0.75`
- task strength：
  - `alpha_min=0.3`
  - `alpha_max=0.9`
  - `alpha_schedule=linear`
- intra-task distill：
  - `teacher_warmup_epochs=5`
  - `beta_distill=0.5`
  - `tau_distill=2.0`

## 7. 实验命令

### 7.1 历史 baseline A

```bash
python -u main.py \
  --schedule 30 60 80 \
  --reg_coef 100 \
  --model_lr 1e-4 \
  --head_lr 1e-3 \
  --svd_lr 5e-5 \
  --bn_lr 5e-4 \
  --svd_thres 10 \
  --model_weight_decay 5e-5 \
  --agent_type svd_based \
  --agent_name svd_based \
  --dataset CIFAR100 \
  --model_optimizer Adam \
  --force_out_dim 0 \
  --first_split_size 10 \
  --other_split_size 10 \
  --batch_size 32 \
  --model_name resnet18 \
  --model_type resnet
```

### 7.2 SFCL baseline B

```bash
python -u main.py \
  --config configs/sfcl_resnet18_cifar100_10.yaml \
  --experiment_name sfcl_full_seed0_v2
```

### 7.3 新方法 C

```bash
python -u main.py \
  --config configs/sfcl_adns_resnet18_cifar100_10.yaml \
  --experiment_name sfcl_adns_full_seed0_v2
```

## 8. ACC / BWT 对比

| 实验 | 类型 | schedule | ACC | BWT | 说明 |
|---|---|---:|---:|---:|---|
| `nscl_historical_baseline` | 历史 baseline | 80 epoch | 73.37 | -1.6444 | 来自 `train_thres_10.log` |
| `sfcl_quick10_seed0` | 新跑 quick baseline | 1 epoch/task | 43.17 | -1.0111 | `sfcl` 快速验证 |
| `sfcl_adns_quick10_seed0` | 新跑 quick method | 1 epoch/task | 48.37 | -9.6444 | `sfcl_adns` 快速验证，含 shared/task-strength/distill |
| `sfcl_full_seed0_v2` | 新跑正式实验 | 80 epoch | 进行中 | 进行中 | 完整长跑 |
| `sfcl_adns_full_seed0_v2` | 新跑正式实验 | 80 epoch | 进行中 | 进行中 | 完整长跑 |

- 当前可见的 quick-run 结果显示：
  - `sfcl_adns_quick10_seed0` 相比 `sfcl_quick10_seed0`，ACC 提升约 `+5.20`
  - 但 quick-run 的 BWT 更负，说明在 1 epoch/task 的极短训练下，增强稳定性尚未转化为更好遗忘控制
- 图表：
  - `acc_bwt_compare.png`
  - `alpha_schedule.png`

## 9. 历史 baseline 与新跑实验区分

- `A / nscl`：
  - 来自历史日志，需明确标记为“历史 baseline”
- `B / sfcl`
  - 当前代码新跑
- `C / sfcl_adns`
  - 当前代码新跑

## 10. 失败项与不确定项

- 早期正式 run 曾因 argparse 默认值覆盖 yaml 而被中止，后已修复并重新启动
- task-level resume 在 smoke checkpoint 上已验证通过
- `auto_select_gpu` 早期会覆盖显式 `--gpuid`，后已修复为“仅在 `gpuid<0` 时自动选卡”
- `sfcl_quick10_seed0` 与 `sfcl_adns_quick10_seed0` 属于快速验证，不应直接与 80-epoch 历史 baseline 做绝对数值公平比较
- 正式 10-task 长实验仍需等待最终 `summary.json`

## 11. 后续建议

- 在正式结果稳定后，再补：
  - `sfcl + shared_lowrank`
  - `sfcl + task_strength`
  - `sfcl + intra_task_distill`
