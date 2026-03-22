# quant-impl

基于 `/project/autoresearch` 最终研究结论落地的 A 股量化实施仓库。

这个仓库不是研究 runner，而是面向实施和日常运行的量化工程版本。它把研究期已经确认的核心协议固定下来，并补齐了生产侧真正需要的几条链路：

- 数据获取：主入口是 `PYTHONPATH=src python -m quant_impl.cli download`，底层复用 [`download_stock.py`](./download_stock.py)，默认单线程、低速率、可续传。
- 数据整理：将单票 parquet 合并为市场级 parquet，再构建训练和推理共用的 bundle。
- 模型训练：保留“线性 IC 先验 + gated residual + shortlist rerank”的主干结构。
- 每日任务：增量抓数、面板刷新、预测归档、历史预测回填验证。
- 统一日志：下载、合并、prepare、train、predict、validate、daily 全部接入统一日志目录和命令级日志文件。

## 固定协议

这里固定的不是某一版参数，而是整套标签和评估口径：

- 标签：`open[t+2] / open[t+1] - 1`
- 交易口径：`t` 日收盘后选股，`t+1` 开盘买入，`t+2` 开盘卖出
- 评估方式：横截面 `top1`
- 数据切分：`5y train + 1y valid + 1y holdout`
- 特征：固定纯日频价量特征，不在日常训练流程里随意改动

## 仓库结构

- `download_stock.py`
  - 东财日线抓取脚本，输出单票 parquet
- `configs/default.yaml`
  - 默认配置文件，已带注释
- `src/quant_impl/cli.py`
  - 所有主命令入口
- `artifacts/logs/`
  - 默认日志目录，自动生成 `download.log`、`train.log`、`daily.log` 等
- `src/quant_impl/data/market.py`
  - 合并市场 parquet、特征工程、bundle、评估口径
- `src/quant_impl/pipelines/train.py`
  - 训练、walk-forward、模型落盘
- `src/quant_impl/pipelines/predict.py`
  - 推理和预测归档
- `src/quant_impl/pipelines/validate.py`
  - 已归档预测的历史验证
- `src/quant_impl/pipelines/daily.py`
  - 每日任务总控
- `tests/`
  - 单元测试和小型 smoke 测试

## 安装

### 1. Python 版本

要求：

- Python `>=3.10`

### 2. 安装依赖

如果你使用当前仓库本地环境，至少需要这些包：

```bash
pip install -e .
```

或者不做 editable 安装，也可以直接：

```bash
pip install numpy pandas pyarrow python-dotenv pyyaml requests torch
```

如果你希望使用下载脚本中的交易日历接口，还需要：

```bash
pip install akshare
```

说明：

- 没有 `akshare` 时，`download_stock.py` 会退化为工作日历近似模式，不会完全阻塞。
- 当前 CLI 不强依赖 `pytest`，测试默认用 `unittest`。

## 两种典型使用方式

### 方式一：你已经有单个市场 parquet

如果你已经在别处准备好了市场级 parquet，比如参考仓库里的：

`/gemini/platform/public/aigc/cv_banc/zsw/zhuangcailin/project/autoresearch/market_daily.parquet`

那么你可以在配置里把 `paths.reference_merged_parquet` 指向这个文件，然后直接训练和预测，不必先抓数。

### 方式二：从单票数据开始

这是默认工程路径：

1. 用 `PYTHONPATH=src python -m quant_impl.cli download` 抓到 `data/daily/*.parquet`
2. 运行 `merge` 合成 `data/market_daily.parquet`
3. 运行 `prepare` 构建 bundle
4. 运行 `train`
5. 运行 `predict`
6. 周期性运行 `validate`

## 快速开始

### 0. 先下载数据

```bash
PYTHONPATH=src python -m quant_impl.cli download
```

这是日常入口，不建议直接把 [`download_stock.py`](./download_stock.py) 当主工作流命令来跑。

默认输出：

- 单票 parquet：`data/daily/*.parquet`
- 下载日志：`artifacts/logs/download.log`
- 下载逐票报告：`artifacts/logs/download_report.jsonl`

### 1. 构建 bundle

```bash
PYTHONPATH=src python -m quant_impl.cli prepare --force
```

作用：

- 读取 `data/market_daily.parquet`，或者配置中的 `reference_merged_parquet`
- 计算固定特征
- 生成训练和推理共用的 bundle

### 2. 训练模型

```bash
PYTHONPATH=src python -m quant_impl.cli train --profile screen --device cuda:3
```

推荐：

- 快速试跑用 `screen`
- 中等检查用 `probe`
- 正式训练用 `full`

### 3. 生成最新预测

```bash
PYTHONPATH=src python -m quant_impl.cli predict --device cuda:3
```

输出：

- `artifacts/predictions/<archive_id>/prediction.json`
- `artifacts/predictions/<archive_id>/top_candidates.csv`

### 4. 回填历史验证

```bash
PYTHONPATH=src python -m quant_impl.cli validate
```

输出：

- `artifacts/validation/history.csv`

### 5. 运行每日流程

```bash
PYTHONPATH=src python -m quant_impl.cli daily --device cuda:3
```

如果需要每日任务时一起重训：

```bash
PYTHONPATH=src python -m quant_impl.cli daily --device cuda:3 --retrain --profile screen
```

## CLI 命令说明

主入口：

```bash
PYTHONPATH=src python -m quant_impl.cli <command> [args]
```

如果做了 `pip install -e .`，也可以直接：

```bash
quant-impl <command> [args]
```

所有子命令都支持这两个统一日志参数：

- `--log-level`
  - 覆盖配置文件中的全局日志级别，例如 `DEBUG`
- `--log-file`
  - 覆盖默认日志路径；不传时默认写 `artifacts/logs/<command>.log`

例如：

```bash
PYTHONPATH=src python -m quant_impl.cli train --profile screen --device cuda:3 --log-level DEBUG
PYTHONPATH=src python -m quant_impl.cli daily --device cuda:3 --log-file artifacts/logs/nightly.log
```

### `download`

```bash
PYTHONPATH=src python -m quant_impl.cli download
PYTHONPATH=src python -m quant_impl.cli download --log-level DEBUG
```

作用：

- 按配置调用 [`download_stock.py`](./download_stock.py)
- 输出到 `paths.raw_daily_dir`
- 默认写 `artifacts/logs/download.log`
- 默认写 `artifacts/logs/download_report.jsonl`

常见配套配置在 `download` 段：

- `start_date`
- `end_date`
- `max_workers`
- `request_interval`
- `request_jitter`
- `retry_sleep`
- `symbols_file`
- `extra_symbols`
- `report_file`

最常用的下载命令就是：

```bash
PYTHONPATH=src python -m quant_impl.cli download
```

如果想更谨慎地观察是否被拒，建议：

```bash
PYTHONPATH=src python -m quant_impl.cli download --log-level DEBUG
```

然后重点看：

- `artifacts/logs/download.log`
  - 请求节奏、重试、疑似反爬信号
- `artifacts/logs/download_report.jsonl`
  - 每只股票最终是成功、空返回还是疑似被拒

### `merge`

```bash
PYTHONPATH=src python -m quant_impl.cli merge --force
```

作用：

- 将 `data/daily/*.parquet` 合成 `data/market_daily.parquet`

### `prepare`

```bash
PYTHONPATH=src python -m quant_impl.cli prepare --force
PYTHONPATH=src python -m quant_impl.cli prepare --limit-stocks 200
```

参数：

- `--force`
  - 强制重建 bundle
- `--limit-stocks`
  - 只处理前 N 只股票，用于 smoke 或调试

### `train`

```bash
PYTHONPATH=src python -m quant_impl.cli train --profile full --device cuda:3
```

参数：

- `--profile`
  - `screen`：快速试跑，fold 和 epoch 都更少
  - `probe`：中等规模检查
  - `full`：完整训练
- `--device`
  - 如 `cpu`、`cuda:0`、`cuda:3`
- `--force-prepare`
  - 训练前强制重建 bundle
- `--limit-stocks`
  - 只训练部分股票，用于调试

训练输出：

- 模型文件：`artifacts/models/*.pt`
- 指标文件：`artifacts/metrics/*_training.json`
- 日志文件：`artifacts/logs/train.log`

### `predict`

```bash
PYTHONPATH=src python -m quant_impl.cli predict --device cuda:3 --as-of-date 2024-08-23
```

参数：

- `--device`
  - 推理设备
- `--as-of-date`
  - 指定打分日期
- `--limit-stocks`
  - 只对部分股票打分

输出：

- 预测归档：`artifacts/predictions/<archive_id>/`
- 日志文件：`artifacts/logs/predict.log`

### `validate`

```bash
PYTHONPATH=src python -m quant_impl.cli validate
```

作用：

- 找到已归档但尚未验证的预测
- 用 market bundle 中对应日期的真实标签回填
- 默认写 `artifacts/logs/validate.log`

### `daily`

```bash
PYTHONPATH=src python -m quant_impl.cli daily --device cuda:3
```

参数：

- `--device`
  - 训练和推理使用的设备
- `--retrain`
  - 当日流程中是否重训
- `--profile`
  - 若重训，使用哪种训练规模

默认流程：

1. 从最近数据日期向前回退若干天做增量抓取
2. 抓取单票 parquet
3. 合并成单个市场 parquet
4. 构建 bundle
5. 可选重训
6. 生成新预测归档
7. 验证旧预测

默认日志：

- 主流程：`artifacts/logs/daily.log`
- 抓数阶段：`artifacts/logs/download.log`

## `download_stock.py` 使用说明

这个脚本现在是底层实现和补充调试入口，不是主入口。日常建议先用：

```bash
PYTHONPATH=src python -m quant_impl.cli download
```

如果你确实想单独控制抓数，不经过主 CLI，可以直接运行：

```bash
python download_stock.py --parquet-dir data/daily
```

常用参数：

```bash
python download_stock.py \
  --parquet-dir data/daily \
  --start-date 2020-01-01 \
  --max-workers 1 \
  --host-max-workers 1 \
  --request-interval 0.8 \
  --request-jitter 0.2 \
  --retry-sleep 1.5 \
  --log-level INFO \
  --log-file artifacts/logs/download.log \
  --no-console-log \
  --report-file artifacts/logs/download_report.jsonl \
  --symbols-file my_symbols.txt
```

### 下载参数说明

- `--start-date`
  - 起始日期，格式 `YYYY-MM-DD`
- `--end-date`
  - 结束日期，默认自动取最近交易日
- `--parquet-dir`
  - 单票 parquet 输出目录
- `--max-workers`
  - 下载线程数，默认建议保持 `1`
- `--host-max-workers`
  - 本机线程上限，默认 `1`
- `--max-retries`
  - 单请求最大重试次数
- `--timeout`
  - 单请求超时时间
- `--request-interval`
  - 每次请求之间的最小间隔秒数
- `--request-jitter`
  - 随机额外等待，降低固定节奏访问风险
- `--retry-sleep`
  - 请求失败后的基础回退秒数
- `--force`
  - 强制覆盖重下
- `--limit`
  - 仅下载前 N 只股票
- `--shuffle-symbols`
  - 打乱股票顺序，降低连续访问模式
- `--log-level`
  - 日志级别，排查问题时建议用 `DEBUG`
- `--log-file`
  - 额外写入文件日志，适合长时间抓数
- `--no-console-log`
  - 关闭终端输出，只保留文件日志
- `--report-file`
  - 每只股票一行 jsonl 结果，最适合排查“被拒绝/空返回/失败”
- `--symbols`
  - 命令行直接传股票列表
- `--symbols-file`
  - 从文件读取股票列表

排查抓数问题时，推荐这样跑：

```bash
python download_stock.py \
  --parquet-dir data/daily \
  --start-date 2020-01-01 \
  --max-workers 1 \
  --host-max-workers 1 \
  --request-interval 1.2 \
  --request-jitter 0.3 \
  --retry-sleep 2.0 \
  --log-level DEBUG \
  --log-file artifacts/logs/download_debug.log \
  --report-file artifacts/logs/download_report.jsonl
```

重点看：

- `download_debug.log`
  - 请求失败的 attempt、状态码、退避等待、疑似反爬标记
- `download_report.jsonl`
  - 每只股票最终是 `downloaded`、`empty`、`rejected_or_empty` 还是 `failed`

### 为什么默认先落单票 parquet

当前实现默认不是“直接下载为单个大 parquet”，而是：

1. 先下载成单票 parquet
2. 再合并成单个市场 parquet

原因：

- 更适合断点续传
- 某只股票失败不影响全部文件
- 增量更新简单
- 在反爬环境下更稳

最终训练和推理阶段，仍然会使用单个市场 parquet。

## 配置文件说明

默认配置文件在：

- [`configs/default.yaml`](./configs/default.yaml)

配置分成七段：

- `paths`
  - 输入输出路径
- `logging`
  - 全局日志行为，控制是否落文件、日志级别、默认日志路径
- `download`
  - 抓数行为
- `data`
  - 特征、标签、切分、评估协议
- `model`
  - 模型结构
- `training`
  - 训练超参数与损失权重
- `inference`
  - 推理和归档行为

推荐的最小改动项通常只有这些：

- `paths.reference_merged_parquet`
- `logging.level`
- `download.symbols_file`
- `download.start_date`
- `download.max_workers`
- `paths.*`
- `inference.prediction_name`

## 常见使用场景

### 场景一：直接复用参考仓库的大 parquet

做法：

1. 在 `configs/default.yaml` 中设置 `paths.reference_merged_parquet`
2. 运行 `prepare`
3. 运行 `train`
4. 运行 `predict`

### 场景二：先小样本 smoke

```bash
PYTHONPATH=src python -m quant_impl.cli prepare --force --limit-stocks 50
PYTHONPATH=src python -m quant_impl.cli train --profile screen --device cuda:3 --limit-stocks 50
```

适合：

- 验证环境
- 看通路是否正确
- 避免一开始就跑全市场长窗口

### 场景三：每日自动运行

```bash
PYTHONPATH=src python -m quant_impl.cli daily --device cuda:3 --retrain --profile screen
```

适合：

- 每日收盘后增量更新
- 自动生成归档预测
- 自动回填旧预测

## 输出文件说明

### 模型

- `artifacts/models/<cache_version>.pt`

内容：

- 模型权重
- 特征名
- data/model/training 配置快照
- 训练摘要

### 训练摘要

- `artifacts/metrics/<cache_version>_training.json`

内容：

- `research_score`
- `cv_valid_*`
- `cv_holdout_*`
- 每个 fold 的摘要

### 预测归档

- `artifacts/predictions/<archive_id>/prediction.json`
- `artifacts/predictions/<archive_id>/top_candidates.csv`

内容：

- 预测日期
- 预计入场和出场日期
- top 候选列表
- 已选股票

### 验证历史

- `artifacts/validation/history.csv`

内容：

- 预测日期
- 选中股票
- 实际收益
- alpha
- 是否命中

## 测试

运行全部测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前测试覆盖：

- bundle 构建
- 训练 smoke
- 预测和验证归档
- 日志初始化和下载参数透传
- 下载器的续传/跳过逻辑

## 注意事项

- 默认抓数参数已经偏保守，先不要急着提高并发。
- `train full` 在真实数据上会比较慢，这是正常的。
- 如果只是验证链路，优先用 `screen + --limit-stocks`。
- 当前训练实现优先保证协议一致和工程可维护性，不是极限性能版本。

## 建议的第一轮命令

如果你已经有参考仓库的市场 parquet，推荐直接这样开始：

```bash
PYTHONPATH=src python -m quant_impl.cli prepare --force --limit-stocks 200
PYTHONPATH=src python -m quant_impl.cli train --profile screen --device cuda:3 --limit-stocks 200
PYTHONPATH=src python -m quant_impl.cli predict --device cuda:3
PYTHONPATH=src python -m quant_impl.cli validate
```
