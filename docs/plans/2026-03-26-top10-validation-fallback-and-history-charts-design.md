# Top10 Validation Fallback And History Charts Design

**Problem:** 当前验证回填只计算模型 Top1 的收益，历史页和详情页也只展示 Top1 结果。这会把“Top1 开盘涨停无法成交”的情况错误记为策略结果，同时无法查看 Top10 每一名的真实收益与可交易状态。

**Decision:** 保持模型排序输出不变，新增“实际执行结果”口径。验证阶段按 Top10 候选从高到低寻找第一个 `open_limit_day1 == false` 的标的作为实际执行；如果 Top10 全部开盘涨停，则当日策略收益记为 `0`。页面同时展示“模型 Top1”和“实际执行结果”，并在历史页增加每日收益与累计净值两张图。

## Scope

- 扩展 `validate_pipeline()`，按 Top10 候选回填候选级真实收益和实际执行结果。
- 扩展 canonical prediction archive，使 `daily/*.json` 保留每个候选的验证信息。
- 扩展 `index.json` / `latest.json` / `history.csv` 的执行摘要字段。
- 改造 `/`, `/history`, `/predictions/[date]` 三个页面。
- 为历史页新增服务端 SVG 图表，不引入新前端依赖。
- 增加 Python 和 Web 回归测试。

## Non-Goals

- 不改变模型打分和 `predict` 生成的原始 Top 排名。
- 不改变训练逻辑和目标构造。
- 不引入客户端图表库或前端 hydration。

## Validation Semantics

### 原始排序 vs 实际执行

- `selected_code` / `selected_score` 继续表示模型原始 Top1。
- `validation.executed_code` / `executed_rank` / `executed_score` 表示实际可成交的执行结果。
- 页面默认把收益、Alpha、净值都解释为“实际执行口径”，不再默认等同于模型 Top1。

### 候选回填

对 `top_candidates[:10]` 中每一名候选：

- 根据 `realized_day_detail_lookup()` 回填：
  - `ideal_return`
  - `strict_open_return`
  - `open_limit_day1`
  - `one_word_day1`
  - `tradeable`
  - `executed`
- 当日最终执行逻辑：
  - 若 Rank 1 可交易，则执行 Rank 1
  - 若 Rank 1 不可交易，则向下顺延到第一个可交易候选
  - 若 Top10 全部不可交易，则收益为 `0`

### 顶层验证摘要

`validation` 新增或强化以下字段：

- `selected_return`: 实际执行收益
- `selected_ideal_return`: 实际执行标的的理想收益；若无可执行标的则为 `0`
- `alpha`: `selected_return - universe_return`
- `executed_code`
- `executed_rank`
- `executed_score`
- `fallback_applied`
- `all_top10_blocked`
- `tradeable`

### 回填版本

为避免旧 `validated` 归档因为顶层收益相同而跳过重写，新增 `validation.schema_version`。当 schema version 不匹配时，即使此前已验证，也要重写 canonical daily payload、legacy run archive 和索引视图。

## Archive Changes

### `daily/YYYY-MM-DD.json`

- `top_candidates` 每条记录新增候选级 `validation`
- `validation` 新增执行摘要
- `summary` 继续从 `validation` 派生，但使用新的实际执行收益口径

### `index.json`

每条记录新增：

- `executed_code`
- `executed_rank`
- `fallback_applied`
- `all_top10_blocked`

### `history.csv`

每条记录新增：

- `executed_code`
- `executed_rank`
- `executed_score`
- `fallback_applied`
- `all_top10_blocked`
- `schema_version`

## Website Design

### `/history`

- 页面顶部新增两张图卡：
  - 每日收益图：展示每个交易日的 `selected_return`
  - 历史净值图：第一天固定为 `1.0`，之后按 `nav *= 1 + selected_return`
- 图表采用服务端内联 SVG 生成，保持静态站点部署简单。
- 历史卡片主标题改为“实际执行标的”；若发生顺延，附加提示如 `Top1 blocked -> Rank 3`。

### `/predictions/[date]`

- 左卡显示模型 Top1 与原始得分。
- 右卡显示实际执行摘要，包括执行标的、执行排名、收益、Alpha、是否顺延、是否 Top10 全部不可交易。
- `Top 候选` 改为表格型布局，列出：
  - Rank
  - 股票
  - 模型分数
  - 实际收益
  - 理想收益
  - 开盘涨停
  - 是否执行

### `/`

- 最新预测摘要沿用 `summary.selected_return` 和 `summary.alpha`，因为它们已经切换到实际执行口径。
- 最近六次历史卡片主展示切换为 `executed_code` 优先，保留对模型 Top1 与顺延的文案说明。

## Testing

### Python

- Top1 开盘涨停时应自动顺延到 Top2，且 `executed_rank == 2`
- Top10 全部开盘涨停时，`selected_return == 0`
- `top_candidates` 每一项都带候选级验证字段
- `history.csv` 和 `index.json` 写出执行摘要字段

### Web

- 新增图表数据聚合与净值计算测试
- 新增 detail/index/history 对 `executed_*` 和候选级验证字段的读取测试
- 保持 `latest.json` 缺失或空文件时的空态行为
