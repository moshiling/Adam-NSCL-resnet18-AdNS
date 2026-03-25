**Change Log**
- 将 shared-subspace 默认模式从旧的并集压缩思路切到 `overlap_core`，同时保留 `union_lowrank` fallback。
- 为 `sfcl` / `sfcl_adns` 补全更细的 scale / boost / projector norm 统计。
- 确认并保留 `sfcl_adns` 的 soft projector 主干，不退化为硬投影。
- 将 `scheduler.step()` 移到 epoch 末尾，修正 baseline 日志里暴露出的不规范调用。
- 将 `teacher_warmup_metrics_task_XX.json`、`train_loss_breakdown.jsonl` 接入主流程。
- 补齐 `shared_subspace_mode`、`shared_overlap_threshold` 等配置，并写入 `config_resolved.json`。
- 新增 [subset_smoke.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/tools/subset_smoke.py) 用于快速 task-subset smoke / resume 检查。
