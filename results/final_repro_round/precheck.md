**Precheck**
实际可用项目根目录为 `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`。用户给出的无空格路径不存在。

baseline 日志实际可用路径为 `/home/moshiling/Adam-NSCL-resnet18+AdNS /train_thres_10.log`。

**1. 已存在相关实验目录**
- `results/method_alignment_fix_round/runs/`
  - `method_alignment_smoke_sfcl`
  - `method_alignment_smoke_sfcl_adns`
  - `method_alignment_smoke_sfcl_v2`
  - `method_alignment_smoke_sfcl_v3`
  - `method_alignment_smoke_sfcl_limited2`
  - `method_alignment_smoke_sfcl_subset`
  - `method_alignment_smoke_sfcl_subset2`
  - `method_alignment_smoke_sfcl_adns_v3`
  - `method_alignment_smoke_sfcl_adns_subset`
  - `method_alignment_smoke_sfcl_adns_subset2`
- `results/adns_sfcl_port/runs/`
  - `sfcl_full_seed0_v2`
  - `sfcl_adns_full_seed0_v2`
  - `sfcl_quick10_seed0`
  - `sfcl_adns_quick10_seed0`
  - `smoke_resume_check_v3`

**2. 完成度核查**
- 完整完成:
  - `method_alignment_smoke_sfcl_v2`
  - `method_alignment_smoke_sfcl_v3`
  - `method_alignment_smoke_sfcl_limited2`
  - `method_alignment_smoke_sfcl_subset`
  - `method_alignment_smoke_sfcl_subset2`
  - `method_alignment_smoke_sfcl_adns_v3`
  - `method_alignment_smoke_sfcl_adns_subset`
  - `method_alignment_smoke_sfcl_adns_subset2`
  - `sfcl_quick10_seed0`
  - `sfcl_adns_quick10_seed0`
  - `smoke_resume_check_v3`
- 不完整:
  - `method_alignment_smoke_sfcl`
  - `method_alignment_smoke_sfcl_adns`
  - `sfcl_full_seed0_v2` 没有 `summary.json`
  - `sfcl_full_seed0`
  - `sfcl_adns_full_seed0`

**3. 当前正在运行的相关进程**
- PID `2563558`
  - `method_alignment_smoke_sfcl_fixed`
  - `python3 tools/subset_smoke.py ... --projection_mode sfcl`
  - 输出目录: `results/final_repro_round/method_alignment_smoke_sfcl_fixed`
  - 属于修正后代码版本: 是
- PID `2563569`
  - `method_alignment_smoke_sfcl_adns_fixed`
  - `python3 tools/subset_smoke.py ... --projection_mode sfcl_adns`
  - 输出目录: `results/final_repro_round/method_alignment_smoke_sfcl_adns_fixed`
  - 属于修正后代码版本: 是
- PID `2563585`
  - `sfcl_full80_seed0_fixed`
  - 输出目录: `results/final_repro_round/sfcl_full80_seed0_fixed`
  - 属于修正后代码版本: 是
- PID `2563596`
  - `sfcl_adns_full80_seed0_fixed`
  - 输出目录: `results/final_repro_round/sfcl_adns_full80_seed0_fixed`
  - 属于修正后代码版本: 是
- PID `2563602`
  - `nscl_full80_seed0_fixed`
  - 输出目录: `results/final_repro_round/nscl_full80_seed0_fixed`
  - 属于修正后代码版本: 是

**4. 状态分类**
状态 `B`
- 修正后 smoke 已有可用完成结果，但标准命名 fixed smoke 之前没有完整落盘
- 修正后正式复现实验尚未完成，因此已进入正式 fixed 主结果阶段

**5. GPU 选择**
- `GPU 0`
  - 空闲显存约 `18.1 GB`
  - 利用率低
  - 适合 `sfcl_full80_seed0_fixed`
- `GPU 3`
  - 空闲显存约 `8.2 GB`
  - 利用率低
  - 适合 `sfcl_adns_full80_seed0_fixed`
- `GPU 5`
  - 空闲显存约 `6.5 GB`
  - 虽不宽裕，但 CIFAR100 + ResNet18 + batch32 足够
  - 用于 `nscl_full80_seed0_fixed`
