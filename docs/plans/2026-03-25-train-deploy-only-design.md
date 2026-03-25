# Train Deploy-Only Design

**Problem:** `quant_impl.cli train` 目前总是先跑整套 walk-forward folds，再在末尾训练 deployment 模型。用户需要一个只训练最终上线模型的入口，避免为产出模型重复跑完整评估。

**Decision:** 保持当前 `train` 默认行为不变，新增显式 CLI 开关 `--deploy-only`。开启后，训练流水线跳过 cross-validation folds，只执行最后一段 deployment 窗口训练，但 deployment 口径调整为“最近 5 年全量训练，不留 valid，不留 holdout，固定跑完配置中的 deployment epoch 后直接保存最后一个 epoch 权重”。

**Scope:**
- 修改 `train` CLI 参数定义和参数透传
- 修改 `train_pipeline()` 支持 deploy-only 模式
- 保持 `daily` 默认行为不变
- 为 deploy-only 增加单元测试

**Behavior:**
- 默认 `train`：继续执行完整 walk-forward + deployment 回训
- `train --deploy-only`：不跑 folds，不生成 CV summary，只训练 deployment 模型
- deploy-only deployment 训练口径：
  - 只取最近 `train_days` 对应的 5 年窗口
  - 不再切出 `valid`
  - 不保留 `holdout`
  - 固定跑完 `training.deployment_epochs`
  - 直接使用最后一个 epoch 的权重，不再基于验证集选 best checkpoint

**Compatibility:**
- 现有默认命令和已有测试语义不变
- deploy-only 返回结构保留 `folds` 和 `summary` 键，其中 `folds=[]`，`summary=None`
- 新增 `deploy_only` 标记，便于日志和产物识别

**Non-Goals:**
- 不改变 `daily --retrain` 的默认 profile
- 不新增“只跑评估不产出模型”模式
- 不改默认完整训练路径的 CV / valid / holdout 口径
