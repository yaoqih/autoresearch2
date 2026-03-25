# Juliang Proxy Refactor Design

## Goal

Remove the existing KDL-specific proxy path and replace it with a Juliang-only proxy flow that:

- extracts one proxy at a time from Juliang,
- reuses that proxy across concurrent downloader requests during its 30-60 second lease,
- keeps Eastmoney cookie warmup optional and disabled by default,
- records whether a failure happened at the proxy transport layer or at the target site layer.

## Context

The current downloader has two independent pieces of network logic:

- a KDL-backed proxy pool inside `download_stock.py`,
- an optional Eastmoney cookie warmer implemented in `tools/eastmoney_cookie_warmer.mjs`.

Recent live experiments against `push2his.eastmoney.com` showed that successful responses do not currently require a warmed cookie. The practical bottleneck is proxy quality and lease utilization, not cookie acquisition. That means the downloader should optimize for reusing one extracted proxy aggressively before rotating.

## Decision

### Proxy Source

Only Juliang remains supported. KDL settings, CLI flags, and provider code are removed.

### Proxy Strategy

The downloader keeps a single active Juliang proxy lease at a time. That lease is shared across request threads. Rotation happens only when:

- the lease is near expiry,
- the proxy transport fails,
- the target reports an auth-style proxy error that indicates the proxy is no longer usable.

This keeps the design simple and matches the user’s package characteristics: one extracted proxy is valid for 30-60 seconds and should be fully utilized before requesting another.

### Cookie Strategy

Eastmoney cookie warmup remains available but becomes opt-in and defaults to off. The downloader should not depend on cookie warmup for normal operation. If enabled, the warmer uses the same active proxy endpoint so the browser session and HTTP requests stay aligned.

### Failure Classification

Request failures are classified into two broad categories:

- `proxy_transport`: TLS, proxy connect, tunnel auth, timeout before a valid target response, or explicit proxy-side status codes like `407`.
- `target_site`: a valid upstream HTTP response or body that indicates Eastmoney-side rejection, throttling, empty-body disconnect patterns, or anti-bot behavior.

This classification must appear in request metadata and downloader logs.

## Configuration

`download` settings become:

- `use_env_proxy`
- `eastmoney_cookie_warmup`
- `eastmoney_cookie_cache_file`
- `eastmoney_cookie_max_age_seconds`
- `eastmoney_cookie_node_binary`
- `eastmoney_cookie_script`
- `eastmoney_browser_path`
- `eastmoney_browser_proxy`
- `eastmoney_cookie_timeout_ms`
- `juliang_enabled`
- `juliang_trade_no`
- `juliang_api_key`
- `juliang_proxy_username`
- `juliang_proxy_password`
- `juliang_num`
- `juliang_proxy_type`
- `juliang_api_base`
- `juliang_lease_refresh_margin_seconds`

Secrets are resolved from `.env` by default. Config only decides whether the Juliang path is enabled and how the lease manager behaves.

## Data Flow

1. `configs/default.yaml` enables Juliang and disables cookie warmup by default.
2. `src/quant_impl/settings.py` loads the new Juliang fields.
3. `src/quant_impl/pipelines/daily.py` forwards Juliang arguments into `download_stock.py`.
4. `download_stock.py` creates a `JuliangProxyManager`.
5. Worker threads borrow the current active proxy from the manager.
6. Requests share that proxy until the lease should rotate or a proxy-layer failure invalidates it.
7. Logs and jsonl reports include failure category details.

## Testing

The implementation is locked with focused unit tests for:

- config loading of Juliang fields,
- CLI plumbing from `run_download_step()`,
- Juliang sign generation and API response parsing,
- single active lease reuse across multiple borrows,
- lease refresh on expiry,
- failure classification between proxy transport and target-site failures,
- cookie warmup remaining optional.
