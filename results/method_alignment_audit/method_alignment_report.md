**Project**
实际项目根目录为 `/home/moshiling/Adam-NSCL-resnet18+AdNS /Adam-NSCL-main`。用户给出的无空格路径不存在，已按要求回退到最接近且真实存在的路径。

baseline 日志实际使用路径为 `/home/moshiling/Adam-NSCL-resnet18+AdNS /train_thres_10.log`。该日志明确显示：
- 数据集设定是 `CIFAR100_10_10`
- backbone 是 `resnet18`
- 优化器是 `Adam`
- `schedule=[30,60,80]`
- `model_lr=1e-4`, `head_lr=1e-3`, `svd_lr=5e-5`, `bn_lr=5e-4`
- `svd_thres=10.0`
- 最终 `ACC=73.37`, `BWT=-1.644444444444441`
- 日志里还直接暴露了旧代码把 `scheduler.step(epoch)` 放在 `optimizer.step()` 前面的 warning

**A-G 审计**
`A. sfcl 是否仍是全空间软投影: YES`
- `build_sfcl_projector()` 使用全部 `eigen_vectors` 构造 `U diag(scale) U^T`，不是只用候选零空间。[projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L67)
- 候选零空间 `basis` 只作为返回值和后续 shared-subspace 输入，不是 projector 的全部基底。[projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L69) [projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L80)

`B. sfcl_adns 是否仍保持 SFCL 主干: YES`
- `build_sfcl_adns_projector()` 先复用 SFCL base scale，再用 `safe_scores` 和 `boosts` 改 `final_scales`，最后仍然是 `U diag(final_scales) U^T`。[projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L100)
- 当前实现没有退化成 `U_share U_share^T` 硬投影。

`C. shared_lowrank 是否真是共享核心: PARTIAL`
- 审计前是典型的 `union_lowrank`，更像拼接后低秩压缩，不像共享核心。
- 本轮已改成默认 `overlap_core`：先算 `M = U_pre^T U_cur`，再按 overlap singular values 和主角方向取 shared basis；旧 `union_lowrank` 保留为 fallback 模式。[subspace_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/subspace_utils.py#L72) [subspace_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/subspace_utils.py#L213)
- 仍是“更接近共享核心”的工程近似，不是论文级唯一标准实现。

`D. rho_t 是否真实参与 projector: YES`
- `_two_stage_scales()` 直接把高奇异值方向缩放乘上 `rho_t`。[projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L25)
- `set_task_context()` 会把 `rho_t` 写入 param group。[adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py#L98)

`E. intra-task distill 是否端到端生效: YES`
- `build_teacher_model()` 在 `train_task()` 中被真实调用。[svd_agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/svd_agent.py#L79) [agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py#L154)
- 第一任务显式禁用 KD。 [agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py#L158)
- `loss = ce_loss + beta_distill * distill_loss` 在训练循环中真实执行。[agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py#L192)
- 本轮新增 `teacher_warmup_metrics_task_XX.json` 与 `train_loss_breakdown.jsonl` 落盘接线。[main.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/main.py#L402)

`F. scheduler / optimizer step 是否合理: NO -> YES`
- baseline 日志已证明旧代码存在 `scheduler.step()` 位置错误 warning。
- 本轮把 `scheduler.step()` 从 batch 循环里移到 epoch 末尾。[agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py#L241)

`G. 是否实现 SFCL 论文中的 SSL/self-supervision 正交增强: NO`
- 全项目搜索未发现 SSL / contrastive / self-supervision 分支。
- 当前实现只代表 soft projection 系列，不代表论文里完整的 SSL 增强版。

**额外确认**
`current_basis = build_sfcl_projector(...)[1]` 的确是当前候选零空间 basis。
- `build_sfcl_projector()` 返回的第二项来自 `compute_candidate_nullspace(...)[0]`。[projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L69) [projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py#L97)
- `Adam.get_transforms()` 里 shared 模块也是用这一返回值作为 `current_basis`。[adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py#L203)

多头结构与 `teacher.last[task_name]` 兼容。
- `model.last` 被重建为 `ModuleDict`，key 就是 task id 字符串。[agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py#L74)

`update_optim_transforms(train_loader)` 的调用时序合理。
- 当前是任务训练完成后，再回放当前任务数据累计 covariance / eigens / transforms，供后续任务使用。[svd_agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/svd_agent.py#L103)

**结论**
当前 `sfcl_adns` 可以准确表述为：
“以 Adam-SFCL 全空间软投影为主干，在 safe/risk 方向上做 shared-overlap-aware scale modulation，并叠加 task-aware strength 与 intra-task KD 的 continual-learning optimizer variant。”
