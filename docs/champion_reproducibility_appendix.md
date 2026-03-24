# Champion Reproducibility Appendix

这份附录补充 `2026-03-23 champion handoff` 中对“策略方案”已经足够、但对“exact reproduction”还不够显式的实现细节。

结论先写在前面：

- handoff 主文档足以指导冠军方案迁移
- 但如果目标是把 full reproduction 和参考结果做到逐项完全一致，仅靠主文档不够
- 下面这些约束必须一起满足，否则很容易出现“策略一样，但最终指标不完全对齐”

## 1. 参考基准

本仓库当前以这组参考对象作为 exact reproduction 基准：

- champion handoff: `/gemini/platform/public/aigc/cv_banc/zsw/zhuangcailin/project/autoresearch/results/runs/20260323_champion_handoff_plan.md`
- canonical result json: `/gemini/platform/public/aigc/cv_banc/zsw/zhuangcailin/project/autoresearch/results/runs/temporal_generalization_full_lc96_results.json`
- reference market parquet: `/gemini/platform/public/aigc/cv_banc/zsw/zhuangcailin/project/autoresearch/market_daily.parquet`

## 2. 冠军方案本体

冠军方案本体仍然是：

- `member_config = lc96`
- `target_transform = rank_center`
- `train_target_abs_cap = 0.10`
- `train_target_cap_applies_to_linear_head = True`
- `temporal_mode = full5y`
- no gate
- final eval trading rule `top1`

这部分决定“你在实现哪条策略线”。

## 3. Exact Reproduction 必须额外满足的约束

下面这些约束决定“你能不能把同一条策略线复现到数值完全一致”。

### 3.1 数据与 prepare 口径必须完全一致

- 使用同一个 `market_daily.parquet`
- 不改 `prepare.py` 的固定特征列、标签定义、切窗长度和评估口径
- 训练阶段的 `train-only cap` 只过滤训练样本，不能扩散到 valid、holdout、inference

### 3.2 fold 与 epoch 的随机种子派生规则必须一致

只写 `seed = 42` 不够，必须把派生规则也固定下来：

- fold seed: `training.seed + split.fold_id`
- deployment seed: `training.seed + len(splits)`
- batch shuffle seed: `seed_base + epoch`

最后这一条很关键。即使别的都一样，只要 batch day 的打乱顺序和参考实现不同，训练轨迹就会逐步漂移。

### 3.3 temporal 路径的 loss 聚合要走 reference 的 weighted 实现

这是这次迁移里最容易漏掉的一点。

参考 temporal 训练虽然在当前 champion 场景下使用的是 `day_weights = 1`，但它实际调用的是 experiment 中的 `weighted_*` loss helper，而不是简单的 unweighted mean 版本：

- `weighted_listwise_rank_loss`
- `weighted_pairwise_rank_loss`
- `weighted_huber_day_loss`
- `weighted_binary_top_loss`
- `weighted_winner_ce_loss`

从数学表达看，两者在 `day_weights = 1` 时等价；但从浮点累计顺序看，它们不是同一条数值路径。神经网络训练对这种微小差异非常敏感，后续 epoch 会逐步放大。

### 3.4 AMP / dtype 行为要贴 reference

需要固定这些行为：

- CUDA 训练使用 `autocast(float16)` 路径
- CPU 路径不应伪造 CUDA AMP，使用 `NoOpGradScaler`
- loss 内部不要引入额外的 in-place blend 或隐式降精度
- label、pos_weight、loss 拼接时的 dtype promotion 要和 reference 路径保持一致

这类问题通常不会在第一个 epoch 就明显爆出来，但会在几轮训练后放大成最终指标偏差。

### 3.5 recent holdout 与 research score 的聚合规则必须一致

除了训练外，评估汇总也必须对齐：

- 最近 `recent_holdout_folds` 个 holdout 在 merge 时按 `recent_holdout_weight = 1.75` 重复采样
- `research_score` 的加权组成固定为：
  - holdout: `0.35`
  - recent: `0.45`
  - valid: `0.20`

如果只看“语义相近”的平均方法，而没有按 reference 的聚合实现来做，最终 summary 也会偏。

## 4. 这次为什么一开始没有对齐

原因不是冠军方案判断错误，而是 handoff 主文档没有把以下“低层但必要”的复现约束写全：

- seed 的派生规则
- temporal loss 实际复用哪套 `weighted_*` helper
- mixed precision 下的 dtype 行为
- summary 聚合的精确路径

所以：

- 报告对“迁移冠军方案”是够的
- 报告对“做 exact reproduction”是不够的

更准确地说，原文是合格的 handoff design note，但不是完整的 bitwise-style reproducibility spec。

## 5. 本仓库中的最终落地

当前实现已经把这些补充约束正式固化到主路径中：

- 默认配置固定为 champion 主线
- 训练阶段正式接入 `rank_center`
- `train-only abs(target) <= 10%` 只作用于训练
- fold seed / deployment seed / epoch shuffle seed 按 reference 规则派生
- temporal 训练 loss 统一走 weighted 聚合路径
- 训练产物写入 `champion_spec`

## 6. 验证结果

已于 `2026-03-24` 完成一次 full reproduction，对照 canonical result json 做逐项比对：

- `research_score`
- `cv_valid_*`
- `cv_holdout_*`
- `recent_holdout_*`

结果：

- 关键指标 `16/16` 与参考 JSON 的绝对误差均为 `0.0`
- `research_score = 0.03741915231302865`

## 7. 给后续维护者的建议

如果后续又要做冠军迁移或重构，建议把“方案定义”和“精确复现约束”分成两个显式层次维护：

- layer 1: champion strategy spec
- layer 2: reproducibility appendix

不要再把 seed 派生、loss 聚合、AMP/dtype 这类隐含约束留在实验脚本或调试过程里。
