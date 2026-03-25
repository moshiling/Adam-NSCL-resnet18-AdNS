# 进度总览

## 已完成

- 已定位真实项目根目录与 baseline 日志路径
- 已从 `train_thres_10.log` 恢复 baseline 核心超参与最终 `ACC/BWT`
- 已完成 `projection_mode=nscl|sfcl|sfcl_adns` 代码接入
- 已补齐：
  - yaml config
  - `config_resolved.json`
  - `summary.json`
  - `task_metrics.jsonl`
  - `plasticity_stats.jsonl`
  - task-level checkpoint / resume
- 已完成 smoke 实验：
  - `smoke_sfcl`
  - `smoke_sfcl_adns`
  - `smoke_sfcl_adns_kd`
- 已完成 resume 验证：
  - `smoke_resume_check_v3`
- 已完成 quick 10-task 实验：
  - `sfcl_quick10_seed0`
  - `sfcl_adns_quick10_seed0`
- 已生成：
  - `experiment_table.csv`
  - `alpha_schedule.png`
  - `acc_bwt_compare.png`

## 正在进行

- 正式实验：
  - `sfcl_full_seed0_v2`
  - `sfcl_adns_full_seed0_v2`

## 待完成

- 汇总正式实验结果
- 生成：
  - 更新版 `final_report.md`
  - 若长跑完成，刷新 `experiment_table.csv`
  - 若长跑完成，刷新 `acc_bwt_compare.png`
