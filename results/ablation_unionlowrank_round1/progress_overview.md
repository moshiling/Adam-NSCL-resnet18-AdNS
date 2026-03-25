**Progress Overview**

- 已完成:
  - `nscl_full80_seed0_fixed_v2` 参考结果
  - `sfcl_full80_seed0_fixed` 参考结果
  - `union_shared_only_full80_seed0`
  - `union_shared_taskstrength_full80_seed0`
  - `union_shared_kd_full80_seed0`
  - `union_shared_taskstrength_kd_full80_seed0` 通过历史正式结果复用
- 正在运行:
  - 无
- 待运行:
  - 无
- 风险说明:
  - E4 使用历史正式结果，不与本轮新跑混写为同一来源
  - E1 曾在 `GPU 0` 停滞，后从 `task_06.pt` 恢复到 `GPU 5` 完成
  - `run.log / run_gpu1.log / resume_task06_gpu5.log` 基本为空，真实有效进度以 `summary.json / task_metrics.jsonl / train_loss_breakdown.jsonl` 为准
