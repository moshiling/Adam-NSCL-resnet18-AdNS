**Precheck**

- 实际项目根目录:
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`
- 用户给定无空格路径不存在；本轮继续使用上面这个真实路径。
- baseline 日志:
  - `/home/moshiling/Adam-NSCL-resnet18+AdNS /train_thres_10.log`
- 数据设定沿用 baseline:
  - dataset=`CIFAR100`
  - split=`10/10`
  - backbone=`resnet18`
  - optimizer=`Adam`
  - schedule=`30 60 80`
  - `model_lr=1e-4`
  - `head_lr=1e-3`
  - `svd_lr=5e-5`
  - `bn_lr=5e-4`
  - `svd_thres=10.0`

**Ablation Status**

- `union_shared_only_full80_seed0`
  - 当前轮新跑
  - 状态: `running`
  - 目录: `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/ablation_unionlowrank_round1/union_shared_only_full80_seed0`
  - 说明: `projection_mode=sfcl_adns`, `shared_subspace_mode=union_lowrank`, `use_task_strength=false`, `use_intra_task_distill=false`
- `union_shared_taskstrength_full80_seed0`
  - 状态: `pending`
  - 原因: 当前资源紧张，按优先级先跑 E1
- `union_shared_kd_full80_seed0`
  - 状态: `pending`
  - 原因: 当前资源紧张，按优先级先跑 E1
- `union_shared_taskstrength_kd_full80_seed0`
  - 状态: `historical_reuse_candidate`
  - 对应历史正式结果:
    - `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/final_repro_round/sfcl_adns_union_lowrank_full80_seed0`
  - 当前配置核对结论: `YES`
  - 说明: 历史 `config_resolved.json` 已确认是 `union_lowrank + use_task_strength=true + use_intra_task_distill=true`

**Reference Runs**

- `nscl_full80_seed0_fixed_v2`
  - 状态: `completed`
  - ACC/BWT: `72.95 / -1.80`
- `sfcl_full80_seed0_fixed`
  - 状态: `completed`
  - ACC/BWT: `74.25 / -4.477777777777779`
- `sfcl_adns_union_lowrank_full80_seed0`
  - 状态: `completed`
  - ACC/BWT: `52.44 / -37.53333333333333`
  - result_source: `historical_formal_result`

**GPU Choice**

- 启动时各卡概况:
  - `GPU 0`: free `22095 MiB`, util `12%`
  - `GPU 2`: free `4581 MiB`, util `0%`
  - `GPU 3`: free `16518 MiB`, util `99%`
  - `GPU 5`: free `17444 MiB`, util `91%`
  - `GPU 6`: free `15525 MiB`, util `100%`
- 本轮优先选择 `GPU 0`
  - 理由: 空闲显存最多且计算利用率最低，更适合正式 80 epoch 新实验
- 由于其余卡计算利用率普遍较高，本轮先串行启动 E1；E2/E3 将在 E1 完成后按优先级继续。
