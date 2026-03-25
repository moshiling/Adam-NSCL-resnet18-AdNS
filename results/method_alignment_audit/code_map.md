**Code Map**
- 训练入口: [main.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/main.py)
- optimizer / 投影主逻辑: [adam_svd.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/adam_svd.py)
- projector builder: [projection_builder.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/optim/projection_builder.py)
- subspace utils: [subspace_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/subspace_utils.py)
- teacher / 训练循环 / scheduler: [agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/agent.py)
- task 训练调度 / covariance 累积: [svd_agent.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/svd_agent/svd_agent.py)
- distillation utils: [distill_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/distill_utils.py)
- alpha / rho schedule: [schedule_utils.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/utils/schedule_utils.py)
- subset smoke helper: [subset_smoke.py](/home/moshiling/Adam-NSCL-resnet18+AdNS%20/Adam-NSCL-main/tools/subset_smoke.py)

**Key Mapping**
- NSCL 硬投影: `candidate_nullspace -> basis basis^T`
- SFCL 软投影: `all eigenvectors -> U diag(scale) U^T`
- sfcl_adns: `SFCL base scale -> overlap-aware safe_score -> boosted final scale`
- shared core: `U_pre, U_cur -> overlap matrix M -> SVD -> shared basis`
- KD: `teacher warmup on current head -> online KD in train_epoch`
- checkpoint/resume: `main.save_checkpoint()` + `Agent.serialize_state()` + `Adam.serialize_projection_state()`
