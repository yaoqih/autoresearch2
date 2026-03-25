# Execution-Aware Control Handoff

日期：`2026-03-26`

## 1. 当前默认主线

`autoresearch2` 当前同步的默认方案是研究仓库中已经确认的 execution-aware control 主线，而不是更早的 `2026-03-23 champion`。

固定配置如下：

- 主干结构：`lc96`
- 时间窗：`full5y`
- 训练目标：`exec_fillable_rank_neg1`
- 训练期样本过滤：仅训练阶段剔除 `abs(raw target) > 10%`
- 交易评估：`top1`
- 严格成交评估：若 `t+1` 开盘已涨停，则该日实际实现收益按 `0` 计

## 2. 为什么默认主线不再是旧 champion

旧 champion 的训练目标是 `rank_center`。它会把未来收益高的样本当成强正样本，但没有区分这些收益是否真的可成交。

研究里后来发现，有一类样本会显著抬高离线回测分数，却很难转化成真实收益：

- `t` 日收盘后看起来信号很强
- `t+1` 一开盘就已经涨停
- 真实交易里买不进去
- 如果仍然把这类样本记成高收益正样本，模型会被鼓励去追逐“看起来最好、但无法成交”的名字

因此当前主线做了两个修正：

1. 训练目标修正
   对 `t+1` 开盘已涨停的股票，在训练目标里直接压到 `-1`
2. 评估口径修正
   如果模型最终选中的股票在 `t+1` 开盘涨停，则该日实现收益按 `0` 计算

这两步的目标是一致的：把训练目标和真实可实现收益对齐。

## 3. 训练目标的准确含义

`exec_fillable_rank_neg1` 的日内构造方式如下：

1. 先拿到该交易日横截面的原始标签：`open[t+2] / open[t+1] - 1`
2. 判断每只股票在 `t+1` 开盘是否已经达到涨停价
3. 对不可买入样本，训练目标直接设为 `-1`
4. 对其余可买入样本，按原始标签做横截面排序，再映射到 `[-1, 1]` 的 centered rank

直观上，这相当于告诉模型：

- “先别把买不进去的名字学成 alpha”
- “在真正可成交的股票里，再学习相对排序”

## 4. 严格成交评估口径

当前 repo2 中的主评估口径是 strict executable eval。

对每天的 `top1` 选股：

- 理想收益：直接使用该股票的原始标签 `open[t+2] / open[t+1] - 1`
- 严格实现收益：如果 `t+1` 开盘已涨停，则记为 `0`；否则等于理想收益

实现里还会额外输出一些辅助指标：

- `trade_rate`
  有多少天最终选中的股票是真正可买入的
- `block_rate_open_limit`
  有多少天被 `t+1` 开盘涨停挡住
- `block_rate_one_word`
  有多少天是更强的“一字板”情形
- `ideal_mean_return`
  不考虑可成交限制时的均值收益
- `one_word_mean_return`
  把一字板当作阻塞条件时的辅助口径

这些辅助指标不是主排序标准，但它们能帮助实施侧判断“模型 alpha”与“可成交性损失”分别来自哪里。

## 5. 与旧主线的关系

旧 `2026-03-23 champion` 没有被删除，仍作为历史材料保留，主要用途有两个：

- 做 exact reproduction
- 解释研究是如何从“离线最优”过渡到“可实现收益更一致”的

如果你的目标是跟当前研究结论对齐，应以本文件定义的 execution-aware control 方案为准，而不是继续沿用 `rank_center` 的默认口径。

## 6. 同步到实现仓库后的落地点

本次 handoff 在 `autoresearch2` 中的关键落地点包括：

- `src/quant_impl/data/market.py`
  - 增加 `open_limit_day1` / `one_word_day1`
  - 新增 `exec_fillable_rank_neg1`
  - `top1` 评估切到 strict executable return
- `src/quant_impl/pipelines/train.py`
  - 训练时把 `open_limit_day1` 作为 blocked flag 传入 target transform
  - summary 增加 trade/block/ideal/one-word 指标
  - artifact 中固化 `strict_executable_eval=true`
- `src/quant_impl/pipelines/validate.py`
  - 历史验证使用 strict executable selected return
- `configs/default.yaml`
  - 默认主线改为 execution-aware control
- `README.md`
  - 默认说明改为当前主线

## 7. 当前不作为默认主线的内容

以下内容保留在研究仓库，不进入 repo2 默认实现：

- shortlist rerank 的条件分支
- 各类 no-trade gate
- regime gating
- 仅为研究比较而存在的历史 target 变体

原因很简单：这些分支虽然在研究里有探索价值，但不适合作为实施仓库的默认主线，容易增加复杂度并削弱维护清晰度。
