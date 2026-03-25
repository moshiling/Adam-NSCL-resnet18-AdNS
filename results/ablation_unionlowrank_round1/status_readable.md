**Readable Status**

你如果不想看 `.json`，直接看这个文件就行。

**当前总状态**

- `E1 shared-only`: 已完成
- `E2 task_strength-only`: 已完成
- `E3 KD-only`: 已完成
- `E4 task_strength+KD`: 已有完整历史结果

**最终结果**

| 实验 | ACC | BWT |
|---|---:|---:|
| `E1 shared-only` | 74.36 | -3.5111 |
| `E2 task_strength-only` | 38.88 | -53.8889 |
| `E3 KD-only` | 74.00 | -2.2333 |
| `E4 task_strength+KD` | 52.44 | -37.5333 |

**最重要结论**

- `shared-only` 几乎不比 `sfcl` 差
- `KD-only` 也几乎不比 `shared-only` 差
- 一旦打开 `task_strength`，性能会明显恶化
- `KD` 在有 `task_strength` 的情况下更像补偿，而不是继续加害

**建议你优先看**

- `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/ablation_unionlowrank_round1/final_report.md`
- `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main/results/ablation_unionlowrank_round1/experiment_table.csv`
