# Adam-NSCL -> SFCL/AdNS 代码映射说明

## 0. 项目根目录与日志

- 用户给定路径名末尾缺少一个空格；当前机器上的实际工程根目录为：
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`
- baseline 日志实际存在于：
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /train_thres_10.log`

## 1. baseline 日志恢复出的关键设定

- 数据集设定：`CIFAR100_10_10`
  - `dataset=CIFAR100`
  - `first_split_size=10`
  - `other_split_size=10`
- backbone：
  - `model_type=resnet`
  - `model_name=resnet18`
- 阈值参数：
  - 日志中实际使用的是 `svd_thres=10.0`
  - 兼容别名可记作 `thres=10` / `a=10`
- 优化器与训练超参：
  - `model_optimizer=Adam`
  - `model_lr=1e-4`
  - `head_lr=1e-3`
  - `svd_lr=5e-5`
  - `bn_lr=5e-4`
  - `weight_decay=5e-5`
  - `schedule=[30,60,80]`
  - `gamma=0.5`
  - `batch_size=32`
  - `reg_coef=100`
- baseline 最终结果：
  - final ACC = `73.37`
  - final BWT = `-1.644444444444441`
- baseline 命令格式：
  - 从 `scripts_svd/adamnscl.sh` 和日志内容交叉恢复，CIFAR100_10 对应命令为：
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

## 2. 原始工程关键路径

### 2.1 训练主入口

- `main.py`
- 原始职责：
  - 解析 CLI
  - 构建数据切分
  - 创建 `svd_agent`
  - 逐任务训练并汇总 `acc/bwt`
- 当前扩展后新增职责：
  - 读取 yaml config
  - 输出 `config_resolved.json`
  - 生成 `summary.json`
  - 生成 `task_metrics.jsonl`
  - 生成 `plasticity_stats.jsonl`
  - 任务级 checkpoint / resume

### 2.2 optimizer / Adam-NSCL 梯度投影

- 原始文件：`optim/adam_svd.py`
- 原始逻辑：
  - `get_eigens(fea_in)`：对累计 feature covariance 做 SVD
  - `get_transforms()`：按阈值取小奇异值方向，构造硬投影矩阵
  - `step()`：将 Adam update 右乘投影矩阵
- 当前扩展后：
  - 支持 `projection_mode = nscl | sfcl | sfcl_adns`
  - 保留原 `nscl` 行为
  - 新增 soft projector 构造
  - 新增 shared subspace / alpha_t / rho_t / per-layer grad-norm 统计
  - 支持序列化/恢复 eigens、projectors、shared subspaces

### 2.3 feature covariance 累计与保存

- 累计入口：
  - `svd_agent/svd_based.py`
  - `compute_cov()` 通过 forward hook 读取层输入
- 累计容器：
  - `svd_agent/svd_agent.py`
  - `self.fea_in`
- 具体方式：
  - `Linear`：直接用输入特征均值做 `X^T X`
  - `Conv2d`：先 `F.unfold` 展开卷积感受野，再 reshape 为 2D matrix，最后做 `X^T X`
- 当前扩展后：
  - `feature_covariance` 会随 checkpoint 一并保存/恢复

### 2.4 SVD / 奇异值筛选 / 投影矩阵构造

- 原始位置：
  - `optim/adam_svd.py`
- 当前拆分：
  - `utils/subspace_utils.py`
    - `compute_candidate_nullspace(...)`
    - `compute_shared_lowrank_subspace(...)`
    - `orthonormalize_basis(...)`
    - `maybe_compute_principal_angles(...)`
  - `optim/projection_builder.py`
    - `build_nscl_projector(...)`
    - `build_sfcl_projector(...)`
    - `build_sfcl_adns_projector(...)`

## 3. ResNet18 / classifier / 任务组织

### 3.1 backbone=ResNet18 注册方式

- `models/resnet.py`
- `resnet18()` 返回 `PreActResNet(PreActBlock, [2,2,2,2])`

### 3.2 classifier 组织方式

- `svd_agent/agent.py`
- 创建模型后会把 `model.last` 替换成 `nn.ModuleDict`
- 每个 task 一个 head：
  - `model.last['1']`
  - `model.last['2']`
  - ...
- 因此当前工程默认是多头 continual learning 组织，而不是单头 class-incremental head

### 3.3 task-incremental / class-incremental

- 当前默认主路径是多头 task-incremental
- `incremental_class=True` 时才会启用 `valid_out_dim` 的单头增量掩码逻辑
- 本次 distillation 默认优先接入多头路径，并只对当前任务 head 做 teacher/student 蒸馏

## 4. 当前阈值 a/thres 如何控制 approximate null space

- 原始规则在 `adam_svd.py` 中等价为：
  - 令奇异值按降序为 `sigma`
  - 取满足 `sigma_i <= sigma_min * thres` 的方向
- 即：
  - `thres` 越大，被视为 approximate null space 的方向越多
  - baseline `thres=10` 对应“保留所有不大于最小奇异值 10 倍”的方向

## 5. 哪些层参与投影

- 所有满足以下条件的层都会参与：
  - `hasattr(m, 'weight')`
  - 不匹配 `last*` 分类头
- 实际包括：
  - `conv1`
  - `stage*.conv*`
  - `stage*.shortcut.0`
- BN 参数不参与 SVD 投影，但单独作为 `bn_lr` 参数组训练

## 6. 从 Adam-NSCL 到 Adam-SFCL 的映射

### 6.1 原始 Adam-NSCL

- 输入：累计的 uncentered feature covariance
- 分解：`cov = U diag(sigma) U^T`
- 子空间：取小奇异值方向 `U2`
- 投影：`P_nscl = U2 U2^T`
- 更新：`g_proj = (P_nscl / ||P_nscl||) g`

### 6.2 当前 Adam-SFCL baseline

- 仍然复用同一份 covariance + SVD
- 但不再只用 `U2`
- 统一使用全空间 `U=[U1,U2]`
- 构造 soft projector：
  - `P_sfcl = U diag(scale_all) U^T`
- 两段式缩放：
  - 大奇异值方向 `U1` 用更小 scale
  - 小奇异值方向 `U2` 用更大 scale
- 当前实现默认：
  - `scale(U1)=1/(1+tau1)`
  - `scale(U2)=tau2/(1+tau2)`

## 7. 从 AdNS 到 sfcl_adns 的迁移

### 7.1 Shared Low-Rank Null Space -> Shared Safe Subspace

- 每层维护：
  - `U_pre`：上一次共享安全子空间
  - `U_cur`：当前 small-singular 子空间
- 拼接：
  - `U_cat = [U_pre, U_cur]`
- 低秩近似：
  - `U_share = low_rank(U_cat)`
- 不做硬投影替换
- 当前 soft-friendly 融合方式：
  - 先在 SVD 基底上得到 `base_sfcl_scale`
  - 再按 `U_share` 与每个奇异向量的 overlap 得到 `safe_score`
  - 最终 `final_scale = base_sfcl_scale * boost(safe_score)`

### 7.2 Non-uniform Constraint Strength -> alpha_t

- 在 `utils/schedule_utils.py` 中实现
- 当前融合：
  - `g_final = (1-alpha_t) * g_raw + alpha_t * g_proj`
- `alpha_t` 支持：
  - `linear`
  - `cosine`
  - `exp`

### 7.3 Intra-task Distillation

- 在 `utils/distill_utils.py` 中实现
- 对 `t>1` 的任务：
  - 冻结 teacher model
  - 只 warmup 当前任务 head
  - 正式训练时加 `KL(student, teacher)`

## 8. checkpoint / summary / log 输出映射

### 8.1 原始工程

- 只有 console log + tensorboard log
- 无显式 checkpoint / resume
- 无 `summary.json` / `task_metrics.jsonl` / `plasticity_stats.jsonl`

### 8.2 当前扩展

- 每个 run 目录输出：
  - `config_resolved.json`
  - `summary.json`
  - `task_metrics.jsonl`
  - `plasticity_stats.jsonl`
  - `checkpoints/last.pt`
  - `checkpoints/task_XX.pt`
  - `tensorboard/*`

## 9. 兼容性结论

- 现有工程是一个非常轻量的 Adam-NSCL 实现，不具备论文工程版的完整实验框架
- 本次改造遵循“最小侵入”原则：
  - 保留原 `svd_based` agent 与 `Adam` optimizer 入口
  - 在 optimizer builder 层新增 `sfcl` / `sfcl_adns`
  - 在 main/agent 外围补齐配置、checkpoint、summary、distillation
- 因此代码映射应以当前工程真实结构为准，而不是强行套论文版完整目录
