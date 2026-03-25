**A. 当前实验状态**
当前状态是 `C`。

已完成：
- `sfcl_full80_seed0_fixed`
- `nscl_full80_seed0_fixed_v2`
- `sfcl_adns_full80_seed0_fixed_v2`
- `sfcl_adns_full80_seed0_fixed_v3_gpu1`
- `method_alignment_smoke_sfcl_fixed`
- `method_alignment_smoke_sfcl_adns_fixed_v3_gpu1`
- `resume_check_shared_subspace_fixed`

未完成或失败：
- `nscl_full80_seed0_fixed`
- `sfcl_adns_full80_seed0_fixed`
- `method_alignment_smoke_sfcl_adns_fixed`

仍在运行：
- `sfcl_adns_union_lowrank_full80_seed0`

被重新启动或替代：
- `nscl_full80_seed0_fixed` -> `nscl_full80_seed0_fixed_v2`
- `sfcl_adns_full80_seed0_fixed` -> `sfcl_adns_full80_seed0_fixed_v2`
- `sfcl_adns_full80_seed0_fixed_v3_gpu1` 作为 GPU1 复现补充
- `method_alignment_smoke_sfcl_adns_fixed` -> `method_alignment_smoke_sfcl_adns_fixed_v3_gpu1`

**B. 修正后代码正式结果表**

| exp_name | mode | shared_subspace_mode | seed | acc | bwt | status | result_source | notes |
|---|---|---:|---:|---:|---:|---|---|---|
| nscl_full80_seed0_fixed_v2 | nscl | overlap_core | 0 | 72.95 | -1.80 | completed | new_run | 当前正式 nscl 主结果 |
| sfcl_full80_seed0_fixed | sfcl | overlap_core | 0 | 74.25 | -4.4778 | completed | new_run | 当前正式 sfcl 主结果 |
| sfcl_adns_full80_seed0_fixed_v3_gpu1 | sfcl_adns | overlap_core | 0 | 49.51 | -40.90 | completed | new_run | 当前正式 sfcl_adns 主结果，且为 GPU1 复现 |

**C. overlap_core vs union_lowrank 消融表**

| exp_name | mode | acc | bwt | status | notes |
|---|---|---:|---:|---|---|
| sfcl_adns_overlap_core_full80_seed0 | sfcl_adns | 49.51 | -40.90 | completed_by_equivalent_config | 使用 `sfcl_adns_full80_seed0_fixed_v3_gpu1` 作为等价 overlap_core 结果 |
| sfcl_adns_union_lowrank_full80_seed0 | sfcl_adns | pending | pending | running | 当前正在跑 |

当前结论：
- 从方法设计上，`overlap_core` 更符合“共享核心”目标
- 但在 `union_lowrank` 结果完成前，还不能给出最终经验结论

**D. 结果解释**
- 修正后版本和历史 baseline 不能做“完全严格公平”的数值对比，因为本轮修正了 `scheduler.step()` 调用位置。
- 因此历史 baseline 只能作为参考，不应被当作和修正后主结果的严格一一对照。
- 当前最可信的主结果是：
  - `nscl_full80_seed0_fixed_v2`
  - `sfcl_full80_seed0_fixed`
  - `sfcl_adns_full80_seed0_fixed_v3_gpu1`
- 当前已经拿到“修正后代码的完整可复现主结果”。
- 当前还没有拿到“shared 子空间结构性修正有效”的完整证据，因为 `union_lowrank` 消融还没跑完。

**E. 下一步建议**
1. 等 `sfcl_adns_union_lowrank_full80_seed0` 完成，给出真正的 `overlap_core vs union_lowrank` 结论。
2. 在现有主结果基础上补 3-seed 稳定性验证，确认 `sfcl` 的提升和 `sfcl_adns` 的极端 BWT 是否稳定。
