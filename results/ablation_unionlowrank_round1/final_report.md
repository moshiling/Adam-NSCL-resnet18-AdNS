**Union Lowrank Round 1**

**A. 本轮实验完成情况**

- 已完成参考结果:
  - `nscl_full80_seed0_fixed_v2`
  - `sfcl_full80_seed0_fixed`
- 已完成本轮新跑:
  - `union_shared_only_full80_seed0`
  - `union_shared_taskstrength_full80_seed0`
  - `union_shared_kd_full80_seed0`
- 已完成可复用历史正式结果:
  - `union_shared_taskstrength_kd_full80_seed0`
    - 复用来源: `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/final_repro_round/sfcl_adns_union_lowrank_full80_seed0`

**B. 四组 union_lowrank 消融结果**

| exp_name | setting | ACC | BWT | source |
|---|---|---:|---:|---|
| `union_shared_only_full80_seed0` | `union_lowrank + shared only` | 74.36 | -3.5111 | resumed |
| `union_shared_taskstrength_full80_seed0` | `union_lowrank + task_strength` | 38.88 | -53.8889 | new_run |
| `union_shared_kd_full80_seed0` | `union_lowrank + KD` | 74.00 | -2.2333 | new_run |
| `union_shared_taskstrength_kd_full80_seed0` | `union_lowrank + task_strength + KD` | 52.44 | -37.5333 | historical_formal_result |

**C. 与 `nscl` / `sfcl` / 完整 union_lowrank 版本的对比**

参考结果:

| exp_name | ACC | BWT |
|---|---:|---:|
| `nscl_full80_seed0_fixed_v2` | 72.95 | -1.80 |
| `sfcl_full80_seed0_fixed` | 74.25 | -4.4778 |

关键对比:

- E1 (`shared only`) vs `sfcl`
  - ACC `+0.11`
  - BWT `+0.9667`
- E1 (`shared only`) vs `nscl`
  - ACC `+1.41`
  - BWT `-1.7111`
- E2 (`task_strength-only`) vs E1
  - ACC `-35.48`
  - BWT `-50.3778`
- E3 (`KD-only`) vs E1
  - ACC `-0.36`
  - BWT `+1.2778`
- E4 (`task_strength + KD`) vs E2
  - ACC `+13.56`
  - BWT `+16.3556`

**D. 对五个问题的逐项回答**

1. 只保留 `union_lowrank shared-subspace` 时，系统是否仍明显差于 `sfcl`？
   - `NO`
   - E1 最终 `74.36 / -3.5111`，与 `sfcl` 的 `74.25 / -4.4778` 非常接近，甚至略优。
   - 这说明 `union_lowrank shared-subspace` 本体并不是性能崩坏的主要来源。

2. `task_strength` 是否是主要伤害项？
   - `YES`
   - E2 相比 E1 发生断崖式恶化：`74.36 / -3.5111 -> 38.88 / -53.8889`
   - 这不仅是 ACC 下降，而且 BWT 也极度恶化。
   - 所以目前最主要的劣化来源就是 `task_strength`。

3. `KD` 是否是主要伤害项？
   - `NO`
   - E3 (`KD-only`) 的结果是 `74.0 / -2.2333`，与 E1 非常接近。
   - 这说明在不启用 `task_strength` 的情况下，KD 没有明显破坏系统，甚至 BWT 还略好于 E1。

4. `task_strength + KD` 是否存在叠加负作用？
   - `不支持“KD 叠加伤害”这个说法`
   - E4 仍然比 E1/E3 差很多，但它明显好于 E2。
   - 这更像是：
     - `task_strength` 先造成主要伤害
     - `KD` 在这个基础上做了部分补偿

5. 当前最合理的下一步方向是什么？
   - 保留 `union_lowrank`
   - 去掉或显著减弱 `task_strength`
   - 不要优先删除 KD

**E. 当前最可能的性能劣化来源判断**

- 最可能的主因: `task_strength`
- 最不支持的说法: “KD 是主要罪魁祸首”
- 当前证据显示:
  - `shared-subspace` 本体可行
  - `KD-only` 基本可行
  - `task_strength` 会导致明显灾难性遗忘

**F. 下一步最值得做的 2 个实验建议**

1. 在 `union_lowrank + shared-only` 基础上，重新引入更弱的 `task_strength`
   - 例如更小的 `alpha_max`
   - 或更平缓的 `alpha_schedule`
2. 固定 `use_task_strength=false`，对 `KD` 做小范围超参搜索
   - 例如 `beta_distill`
   - 或 `tau_distill`

**结论一句话**

本轮已经基本定位清楚：
- `union_lowrank` 不是问题核心
- `KD` 不是主要伤害项
- 真正把系统拉坏的是 `task_strength`
