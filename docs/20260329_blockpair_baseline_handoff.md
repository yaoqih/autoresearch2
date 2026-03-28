# 2026-03-29 Blockpair Baseline Handoff

## 正式主线

部署仓当前正式主线同步为：

- baseline: `exec_fillable_rank_neg1_blockpair_w0p10`
- execution metric: `hybrid`
- execution rule: `top1 + rollover_top5`
- target filter: 仅训练期 `abs(target) <= 10%`
- temporal mode: `full5y`
- deploy epoch: `4`

对应默认命名：

- `data.cache_version = mainline_20260329_lc96_exec_fillable_rank_neg1_blockpair_w0p10_hybrid_rollover_top5_full5y_ep4`
- `inference.prediction_name = mainline_20260329_lc96_exec_fillable_rank_neg1_blockpair_w0p10_hybrid_rollover_top5_full5y_ep4`

## 这次同步改了什么

- 在训练配置里正式加入：
  - `execution_aux_mode = blocked_pairwise`
  - `execution_aux_weight = 0.10`
  - `execution_aux_top_fraction = 0.10`
- 在 `ranker.py` 中加入 blocked-vs-fillable 的 pairwise auxiliary loss。
- 在训练管线里把 `raw_targets` 和 `blocked_flags` 传给 loss，并记录 `train_aux_loss`。
- 在 `champion_spec` 里持久化 auxiliary 配置，避免训练和部署侧出现“模型是这个版本，但 spec 看不出来”的情况。

## 口径约束

- `current / exact` 不再作为 winner 口径。
- `2025` 在研究侧已转为袋外检查集；部署仓这里不负责重新挑 baseline，只负责忠实实现已经冻结的 baseline。

## 同步后建议的验收项

- 单元测试：
  - `tests.test_champion_spec`
  - `tests.test_market_bundle`
  - `tests.test_prediction_validation`
- 最小 smoke：
  - `PYTHONPATH=src python -m quant_impl.cli train --profile screen --device cpu --limit-stocks 200 --force-prepare`
- 关注点：
  - 默认配置名是否带 `blockpair_w0p10`
  - `champion_spec` 是否落盘 `execution_aux_*`
  - 训练摘要里是否出现 `train_aux_loss`
