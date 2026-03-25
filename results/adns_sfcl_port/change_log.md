# 修改清单

## 新增文件

- `configs/sfcl_resnet18_cifar100_10.yaml`
- `configs/sfcl_adns_resnet18_cifar100_10.yaml`
- `optim/projection_builder.py`
- `utils/subspace_utils.py`
- `utils/schedule_utils.py`
- `utils/distill_utils.py`
- `results/adns_sfcl_port/code_map.md`
- `results/adns_sfcl_port/progress_overview.md`

## 重写/扩展的核心文件

- `main.py`
  - 新增 yaml config 读取
  - 新增 `projection_mode` 与新超参 CLI
  - 新增 run 目录、summary、jsonl、checkpoint、resume
  - 新增 GPU 自动选择逻辑
- `optim/adam_svd.py`
  - 从单一 NSCL 投影扩展为 `nscl/sfcl/sfcl_adns`
  - 新增 per-layer projector 构造、shared subspace、task-strength 融合
  - 新增投影状态序列化/恢复
- `svd_agent/agent.py`
  - 新增 alpha/rho 调度接入
  - 新增 intra-task distillation 训练路径
  - 新增 agent 级 state serialize/load
- `svd_agent/svd_agent.py`
  - 新增 projection config 注入 optimizer
  - 新增 feature covariance checkpoint 恢复
  - 新增 projector build stats 输出

## 兼容性说明

- 原始 `Adam-NSCL / svd_based` 入口保留
- `projection_mode=nscl` 保留原硬投影路径
- 所有新功能都通过配置开关控制
- baseline 命令格式仍可继续使用
