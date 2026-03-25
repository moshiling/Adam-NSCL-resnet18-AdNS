**Alignment Conclusion**
当前实现已经从“并集压缩 + 软投影增强”修正为“以 Adam-SFCL 全空间软投影为主干、以 overlap-core shared safe subspace 做 scale modulation、叠加 task-aware strength 与 intra-task KD”的版本。最关键的两处对齐修正是：
- shared subspace 默认改成 `overlap_core`
- scheduler 调用修正为 epoch 级

**A-G 回答**
`A YES` `sfcl` 仍是全空间软投影。  
`B YES` `sfcl_adns` 仍是 SFCL projector 上的 scale 修正。  
`C PARTIAL->IMPROVED` shared subspace 由旧 `union_lowrank` 修正为默认 `overlap_core`。  
`D YES` `rho_t` 真实参与高奇异值方向缩放。  
`E YES` 当前任务 KD 真实进入 `total_loss = ce + beta * kd`。  
`F YES` scheduler 位置已修正。  
`G NO` 没有 SSL/self-supervision 正交增强分支。

**What Was Fixed**
- [subspace_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/subspace_utils.py)
  - 新增 `compute_shared_core_subspace()`
  - `compute_shared_lowrank_subspace()` 默认走 `overlap_core`
- [projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py)
  - 补 `base_scale/safe_score/boost/final_scale/projector_norm` 统计
  - 明确保持 `U diag(scale) U^T`
- [adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py)
  - 新增 `shared_subspace_mode/shared_overlap_threshold`
  - shared basis、eigens、transforms、task context 继续 checkpoint/resume
- [agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py)
  - `scheduler.step()` 改到 epoch 末尾
  - KD 端到端保持生效
- [main.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/main.py)
  - 新配置写入 `config_resolved.json`
  - 新增 `teacher_warmup_metrics_task_XX.json`
  - 新增 `train_loss_breakdown.jsonl`

**What Was Already Correct**
- `sfcl` 本身就是全空间 soft projector，不是 candidate-nullspace-only projector。
- `sfcl_adns` 在本轮修正前也没有硬退化成 `U_share U_share^T`。
- `update_optim_transforms(train_loader)` 的时序本来就是任务训练后更新，用于下一个任务。

**Still Approximate**
- `overlap_core` 是更接近“共享核心”的工程实现，但仍不是唯一论文标准。
- 当前项目没有复现 SFCL 论文中的 SSL/self-supervision 模块。

**How To Name The Method**
推荐论文中写成：
“SFCL-style full-space soft projection with overlap-core shared safe subspace modulation, task-aware projection strength, and intra-task head distillation.”

中文可写成：
“以 SFCL 全空间软投影为主干，融合 overlap-core 共享安全子空间、任务感知软约束强度和任务内蒸馏的优化器方法。”

**Experiments**
- 历史 baseline 已恢复：`ACC 73.37 / BWT -1.6444`
- 上一轮真实 quick run：
  - `sfcl_quick10_seed0`: `43.17 / -1.0111`
  - `sfcl_adns_quick10_seed0`: `48.37 / -9.6444`
- 本轮新代码 smoke / subset-smoke 已启动，结果目录位于 `results/method_alignment_fix_round/runs/`
- 受当前环境中并行长跑与 Python 文件锁竞争影响，本轮部分新跑仍在进行；因此本报告把“源码一致性结论”和“已完成运行证据”分开呈现

**Checkpoint / Resume**
- shared subspaces 的序列化与恢复逻辑仍在 [adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py#L282) 与 [adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py#L306)
- 上一轮 `smoke_resume_check_v3` 已验证 resume 成功；本轮 subset-resume 脚本已补齐，待当前资源空闲即可直接复跑

**Recommended Paper Wording**
- 不要写“复现完整 Adam-SFCL”。
- 应写“实现了 SFCL-style soft projection backbone；未包含论文中的 SSL/self-supervised orthogonal enhancement branch”。
