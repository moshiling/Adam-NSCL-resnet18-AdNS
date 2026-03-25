**Risk List**
- `overlap_core` 仍是工程近似，不是论文唯一标准答案；建议后续做 `overlap_threshold` 与 `topk_ratio` ablation。
- scheduler 位置修正后，和历史 baseline 的数值不再严格逐行可比；对比时必须注明“训练语义修正后结果”。
- 当前没有 SSL/self-supervision 分支，论文表述不能写成“完整 SFCL 复现”。
- 本轮 subset smoke 受环境里并行长实验影响，部分新跑验证处于进行中；需要在资源空闲时补全完整 smoke 表。
