# Eastmoney Cookie 与会话工作原理说明

更新时间：2026-03-22

## 目的

本文档总结本仓库在调试 `push2his.eastmoney.com` 历史 K 线接口时，已经验证过的现象、背后的高层工作原理，以及当前可行的工程方案。

本文档只覆盖：

- 已观察到的请求链路
- Cookie 与页面脚本之间的关系
- 为什么普通 `requests` 会失败
- 哪些方案已经验证可行

本文档不覆盖：

- 复刻或绕过 Eastmoney 访问控制的实现细节
- 仿造受保护指纹流程的代码

## 结论摘要

在 2026-03-22 的实测中，可以确认以下几点：

1. `push2his.eastmoney.com` 这个接口在没有有效会话 Cookie 时，服务端常见行为不是返回 `403` 或验证码页面，而是直接断开连接，表现为：
   - 浏览器：`net::ERR_EMPTY_RESPONSE`
   - `fetch`：`TypeError: Failed to fetch`
   - `requests`：`RemoteDisconnected('Remote end closed connection without response')`

2. 对当前测试样本来说，最关键的 Cookie 是：
   - `nid18`
   - `nid18_create_time`

3. 这两个 Cookie 不是 `https://quote.eastmoney.com/` 首页响应头直接下发的，而是在页面加载后，由页面脚本动态生成并写入浏览器 Cookie。

4. 正常打开 `https://quote.eastmoney.com/` 一段时间后，再把浏览器里的 `nid18` 与 `nid18_create_time` 交给普通 `requests`，就可以成功访问 `push2his`。

5. 因此，当前最可行的工程方案不是“直接裸 `requests` 抓接口”，而是：
   - 先建立一个正常页面会话
   - 获取会话 Cookie
   - 再用普通 HTTP 客户端复用这个会话做下载

## 已验证现象

### 1. 直接请求 `push2his` 会失败

在不带 Cookie 的情况下，请求如下接口：

`https://push2his.eastmoney.com/api/qt/stock/kline/get?...`

会出现：

- Playwright/Chrome `page.goto(...)`：`net::ERR_EMPTY_RESPONSE`
- 浏览器页内 `fetch(...)`：`TypeError: Failed to fetch`
- Python `requests`：连接被远端直接关闭

这说明问题不只是 JS 层面的 CORS 提示，而是服务端在响应头返回之前就终止了连接。

### 2. 浏览器 Cookie 足以让普通 `requests` 成功

在 2026-03-22 的实测中：

- 先用浏览器打开 `https://quote.eastmoney.com/`
- 等待约 8 秒
- 从浏览器上下文读取 `.eastmoney.com` 域下的 Cookie
- 把这些 Cookie 带到普通 `requests`

请求 `push2his` 后返回：

- `HTTP 200`
- `Content-Type: application/json; charset=UTF-8`
- 正常 JSON 正文

进一步缩小后发现，只带下面两个 Cookie 也能成功：

- `nid18`
- `nid18_create_time`

### 3. 首页本身不会直接返回 `nid18`

对 `https://quote.eastmoney.com/` 做纯 HTTP 抓取时：

- 首页是 `200 OK`
- 响应头里没有 `Set-Cookie: nid18`

这说明 `nid18` 不是首页响应头直接发的。

### 4. `nid18` 是前端脚本运行后写入的

在浏览器里拦截 `document.cookie` 写入后，可以看到：

- `quote.eastmoney.com` 页面会加载 `usercollect.min.js`
- `usercollect.min.js` 会进一步动态插入
  `https://anonflow2.eastmoney.com/ewtsdk/ewtsdk.prod.js?ver=20251128`
- 最终由该 SDK 负责写入：
  - `nid18`
  - `nid18_create_time`
  - `gviem`
  - `gviem_create_time`
  - 以及若干统计类 Cookie

## 高层工作原理

可以把当前链路理解成 4 个阶段。

### 阶段 1：页面初始化

用户打开：

`https://quote.eastmoney.com/`

页面会加载自己的业务 JS、统计脚本和会话相关脚本。

### 阶段 2：页面侧脚本收集环境信息

根据已观察到的 SDK 逻辑，高层上它会收集一组浏览器/设备环境信息，例如：

- `userAgent`
- `screenResolution`
- `language`
- `timezone`
- `canvas`
- `webgl`
- `font`
- `audio`

这些信息会被整理为一组摘要值或指纹相关字段。

### 阶段 3：页面建立 Eastmoney 会话标识

在已有 Cookie 不满足条件时，页面侧脚本会建立或刷新一组会话相关标识，最终写回浏览器 Cookie。

从这次实验可以确认，后续访问 `push2his` 至少需要：

- `nid18`
- `nid18_create_time`

### 阶段 4：数据接口校验会话

当客户端请求：

`https://push2his.eastmoney.com/api/qt/stock/kline/get`

服务端会检查当前请求是否带有满足要求的会话上下文。

如果缺失关键会话标识，当前观察到的行为是直接断开连接，而不是返回一个清晰的业务错误码。

## 为什么浏览器能成，而裸 `requests` 经常失败

核心差异不在“浏览器 header 多几个”，而在“浏览器已经跑完了页面脚本，建立了 Eastmoney 自己认可的会话状态”。

具体来说：

1. 浏览器打开 `quote.eastmoney.com`
2. 页面脚本运行
3. 页面脚本写入 `nid18` / `nid18_create_time`
4. 浏览器再去访问 `push2his`

而裸 `requests` 默认直接跳到第 4 步，缺少前面的会话建立过程，因此容易被直接断开。

## 当前可行方案

### 方案 A：浏览器预热一次，下载阶段全走 `requests`

这是当前最推荐的方案。

流程如下：

1. 用正常浏览器上下文打开 `https://quote.eastmoney.com/`
2. 等待数秒，直到会话 Cookie 稳定出现
3. 读取 `.eastmoney.com` 域下 Cookie
4. 取出至少这两个字段：
   - `nid18`
   - `nid18_create_time`
5. 把它们写入本地缓存
6. 后续下载阶段使用普通 `requests`，并附带这两个 Cookie
7. 当 Cookie 失效时，再重新预热一次

优点：

- 下载过程仍然是普通 HTTP 客户端
- 不需要浏览器全程参与抓数
- 与当前观测到的站点行为一致

缺点：

- 仍然需要一个“会话预热”步骤
- Cookie 存在过期与刷新问题

### 方案 B：会话预热器 + Cookie 缓存服务

这是方案 A 的工程化版本。

可以单独做一个小模块，职责只有两件事：

1. 负责建立和刷新 Eastmoney 会话
2. 把最新可用 Cookie 提供给下载器

适合场景：

- 每天定时抓取
- 多个下载任务复用同一份会话
- 希望把“会话建立”和“数据下载”隔离开


## 当前不建议的方向

### 1. 直接裸调 `push2his`

原因：

- 已实测会被直接断连接
- 稳定性不足

### 2. 仅通过补几个浏览器请求头解决问题

原因：

- 已实测，单纯 header 不足以建立可用会话
- 真正起关键作用的是会话 Cookie，而不是表面的 `sec-*` 请求头

### 3. 依赖首页响应头拿 Cookie

原因：

- 已实测首页响应头并不直接返回 `nid18`
- 关键 Cookie 来自页面脚本执行后的客户端写入

## 实施建议

如果继续沿当前仓库实现，建议采用下面的职责拆分：

### 模块 1：Cookie 预热器

职责：

- 打开 `quote.eastmoney.com`
- 等待页面脚本完成会话初始化
- 读取 `nid18` / `nid18_create_time`
- 输出到本地文件或内存对象

### 模块 2：Cookie 存储

职责：

- 保存最近一次成功的会话 Cookie
- 记录获取时间
- 在过期或失效后触发重新预热

### 模块 3：下载器

职责：

- 专注于数据请求与落盘
- 使用现成 Cookie 做请求
- 当发现再次出现“空响应断连”时，回退到 Cookie 刷新流程

## 建议的故障判定规则

对当前接口，可以把下面这些信号视为“会话可能失效，需要重新预热”：

- `RemoteDisconnected`
- `ERR_EMPTY_RESPONSE`
- `Failed to fetch`
- 反复返回空数据且无有效业务错误信息

## 本次结论的边界

本文档的结论来自 2026-03-22 在本机环境上的实测，适用于解释当前看到的问题与当前验证通过的方案。

需要注意：

- 站点前端脚本和 Cookie 机制可能后续变化
- Cookie 有失效时间
- 页面脚本加载顺序和初始化耗时可能变化
- 生产环境还需要补充刷新策略、失败回退和日志

## 下一步建议

建议优先实现：

1. 浏览器预热 Cookie
2. 本地缓存 `nid18` / `nid18_create_time`
3. 下载器自动复用 Cookie
4. 失败时自动重新预热

如果后续要继续工程化，可以再补：

- Cookie 过期判定
- 预热重试
- 下载失败自动刷新会话
- 日志与观测指标
