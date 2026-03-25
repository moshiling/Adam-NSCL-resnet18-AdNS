**Progress Check**
实际项目根目录为 `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`。

**状态分类**
当前属于 `C`：
- 正式主结果已经拿到
- 结构性消融还未全部完成

**正式主结果检查**
- `nscl_full80_seed0_fixed`
  - 不完整，仅有 `config_resolved.json`
  - 不计入最终主结果
- `nscl_full80_seed0_fixed_v2`
  - 完整
  - `ACC=72.95`
  - `BWT=-1.80`
- `sfcl_full80_seed0_fixed`
  - 完整
  - `ACC=74.25`
  - `BWT=-4.4778`
- `sfcl_adns_full80_seed0_fixed`
  - 不完整，仅有 `config_resolved.json`
  - 不计入最终主结果
- `sfcl_adns_full80_seed0_fixed_v2`
  - 完整
  - `ACC=49.51`
  - `BWT=-40.90`
- `sfcl_adns_full80_seed0_fixed_v3_gpu1`
  - 完整
  - `ACC=49.51`
  - `BWT=-40.90`
  - 与 `v2` 一致，可作为更可信的 GPU1 复现实验

**结构性消融检查**
- `sfcl_adns_overlap_core_full80_seed0`
  - 尚未单独命名运行
  - 目前可用等价结果源：`sfcl_adns_full80_seed0_fixed_v3_gpu1`
  - 配置上等价于 `overlap_core`
- `sfcl_adns_union_lowrank_full80_seed0`
  - 已启动
  - 当前仍在运行

**smoke / resume**
- `method_alignment_smoke_sfcl_fixed`
  - 完整，`ACC=19.53125`, `BWT=0.0`
- `method_alignment_smoke_sfcl_adns_fixed`
  - 不完整
- `method_alignment_smoke_sfcl_adns_fixed_v3_gpu1`
  - 完整，`ACC=19.53125`, `BWT=0.0`
- `resume_check_shared_subspace_fixed`
  - 完整，`ACC=19.53125`, `BWT=0.0`

**当前运行进程**
- 当前没有旧的主结果进程在跑
- 当前新增运行：
  - `sfcl_adns_union_lowrank_full80_seed0`

**文件锁 / GPU / 卡死检查**
- 之前存在重复拉起与不完整目录，当前主结果已经通过 `*_v2` / `*_v3_gpu1` 规避覆盖问题
- 当前没有发现仍在运行的失败旧进程
- `union_lowrank` 正在独立运行，GPU 冲突可控

**overlap_core vs union_lowrank 配置一致性**
- `union_lowrank` 使用 `configs/sfcl_adns_resnet18_cifar100_10.yaml`
- 仅额外覆盖 `--shared_subspace_mode union_lowrank`
- 其余保持与当前 `sfcl_adns` 正式结果一致
