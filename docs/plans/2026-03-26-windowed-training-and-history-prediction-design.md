# Windowed Training And History Prediction Design

**Problem:** 当前仓库只能按默认 deployment 口径训练“最近 rolling 5y”模型，并且 `predict` 只支持单日预测。用户需要两条可复用的新链路：

- 用固定时间窗 `2021-01-01` 到 `2025-12-31` 训练 deployment 模型，然后批量推理最近 3 个月，并把每个交易日的真实结果回填到预测归档。
- 用“当前日期向前回看 5 年”的连续时间窗训练 deployment 模型，不留 `valid` 空档，然后预测接下来 1 个交易日。

**Decision:** 保持现有默认 `train` / `predict` / `validate` 行为不变，新增两类能力：

1. `train` 支持显式 deployment 训练时间窗。
2. CLI 新增历史批量预测入口，复用现有归档和验证逻辑。

这样可以避免写一次性脚本，同时保留现有模型契约、预测归档和展示站消费方式。

## Scope

- 扩展 `train` CLI 和 `train_pipeline()`，支持按绝对日期或“锚点日期向前回看 N 年”选择 deployment 训练窗。
- 新增历史批量预测流水线，支持按绝对日期或“锚点日期向前回看 N 月”批量生成单日预测。
- 历史批量预测完成后可直接触发现有 `validate_pipeline()`，把真实结果回填到 `daily/*.json`、`index.json`、`latest.json` 和 `history.csv`。
- 更新 README 和测试。

## Non-Goals

- 不改变现有 walk-forward cross-validation 的 `5y train + 1y valid + 1y holdout` 默认协议。
- 不改变现有单日 `predict` 的输出结构。
- 不引入新的预测归档格式。

## CLI Design

### `train`

保留现有参数，新增以下 deployment 窗口参数：

- `--deployment-start-date YYYY-MM-DD`
- `--deployment-end-date YYYY-MM-DD`
- `--deployment-anchor-date YYYY-MM-DD`
- `--deployment-lookback-years N`

规则：

- 如果给了 `deployment_start_date` 或 `deployment_end_date`，按显式绝对时间窗选取交易日。
- 如果给了 `deployment_anchor_date + deployment_lookback_years`，从锚点日期向前回看 N 个日历年，再对齐到 bundle 中实际存在的交易日。
- 显式 deployment 时间窗只影响最终 deployment 模型，不影响默认 walk-forward folds。
- 当 `--deploy-only` 与显式 deployment 时间窗同时使用时，deployment 训练直接覆盖这个连续窗口，`valid=[]`，不做 best-checkpoint 选择。
- 当没有给任何新参数时，保持现在的默认行为。

### `predict-history`

新增子命令：

```bash
PYTHONPATH=src python -m quant_impl.cli predict-history \
  --start-date 2026-01-01 \
  --end-date 2026-03-26 \
  --validate
```

新增参数：

- `--start-date YYYY-MM-DD`
- `--end-date YYYY-MM-DD`
- `--anchor-date YYYY-MM-DD`
- `--lookback-months N`
- `--validate`
- `--device`
- `--limit-stocks`

规则：

- 必须提供一组范围参数：要么 `start_date + end_date`，要么 `anchor_date + lookback_months`。
- 批量预测会从训练契约对应的数据 bundle 中找出范围内可验证的交易日，对每个交易日调用现有单日预测逻辑写归档。
- `--validate` 打开后，在批量预测结束后调用现有 `validate_pipeline()`，把最近这批历史预测的真实结果回填。

## Data And Time Window Semantics

- 时间窗对齐基于 bundle 中的交易日列表，而不是自然日逐天穷举。
- 显式训练窗和批量历史预测窗都允许用户给出非交易日；实现会自动吸附到窗内最近可用交易日。
- “当前日期往前 5 年训练”对应的命令形态为：

```bash
PYTHONPATH=src python -m quant_impl.cli train \
  --deploy-only \
  --deployment-anchor-date 2026-03-26 \
  --deployment-lookback-years 5
```

- “接下来 1 天预测”仍然使用现有单日 `predict`，默认对最新可用市场日期打分，并输出 `entry_date` / `exit_date`。

## Pipeline Changes

### Training

- 在 `train.py` 中抽出 deployment 时间窗解析函数，负责：
  - 将显式时间窗解析为 bundle day indices。
  - 处理边界吸附和空窗口报错。
  - 在结果和 artifact 中记录 deployment window metadata，便于追溯。

### History Prediction

- 新增 `src/quant_impl/pipelines/predict_history.py`。
- 批量入口内部复用 `predict_pipeline()`，避免复制单日预测归档逻辑。
- 批量返回结果包含：
  - 实际处理的日期范围
  - 生成的预测数量
  - 如启用验证，则附带验证结果摘要

## Error Handling

- 显式训练窗解析后没有任何交易日时，抛出清晰错误。
- `predict-history` 没有提供完整范围参数时，抛出参数错误。
- 历史预测范围内没有任何可预测交易日时，抛出清晰错误，不静默成功。

## Testing

- 为 deployment 显式时间窗增加训练测试。
- 为“锚点日期向前回看 5 年”增加训练测试。
- 为 `predict-history --validate` 增加端到端测试，确认：
  - 会生成多个 `daily/*.json`
  - 会调用验证并把 `status` 变成 `validated`
  - `history.csv` 写入对应条目

## Recommended Workflows

固定 `2021-2025` 训练后推最近 3 个月并回填验证：

```bash
PYTHONPATH=src python -m quant_impl.cli train \
  --deploy-only \
  --deployment-start-date 2021-01-01 \
  --deployment-end-date 2025-12-31

PYTHONPATH=src python -m quant_impl.cli predict-history \
  --anchor-date 2026-03-26 \
  --lookback-months 3 \
  --validate
```

按当前日期向前 5 年连续训练，然后预测下一交易日：

```bash
PYTHONPATH=src python -m quant_impl.cli train \
  --deploy-only \
  --deployment-anchor-date 2026-03-26 \
  --deployment-lookback-years 5

PYTHONPATH=src python -m quant_impl.cli predict
```
