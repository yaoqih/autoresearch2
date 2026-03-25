from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import requests

from download_stock import (
    CrawlerConfig,
    ConsoleProgressBar,
    DownloadPlan,
    EastmoneyCookieWarmupConfig,
    EastmoneyCookieWarmupManager,
    FetchResult,
    HistoricalQuoteClient,
    HttpClient,
    Instrument,
    JuliangProxyConfig,
    JuliangProxyManager,
    OutcomeRecorder,
    OutputTracker,
    ParquetStore,
    ProxyState,
    RequestMeta,
    RequestFailedError,
    StockCrawler,
    TradingCalendar,
    build_juliang_sign,
    parse_dates,
    resolve_worker_count,
    resolve_default_end_date,
)


class DownloaderPendingTest(unittest.TestCase):
    def test_build_juliang_sign_sorts_parameters(self) -> None:
        params = {
            "trade_no": "1483587531995538",
            "num": "1",
            "result_type": "json",
            "pt": "1",
        }
        expected = hashlib.md5(
            "num=1&pt=1&result_type=json&trade_no=1483587531995538&key=demo-key".encode("utf-8")
        ).hexdigest()
        self.assertEqual(build_juliang_sign(params, "demo-key"), expected)

    def test_juliang_proxy_manager_reuses_active_proxy_until_refresh_margin(self) -> None:
        class StubResponse:
            def __init__(self, payload: dict):
                self.payload = payload
                self.status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self.payload

        class StubSession:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict | None]] = []
                self.trust_env = False

            def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> StubResponse:
                del timeout
                self.calls.append((url, params))
                if url.endswith("/dynamic/getips"):
                    index = sum(1 for item, _ in self.calls if item.endswith("/dynamic/getips"))
                    return StubResponse(
                        {
                            "code": 200,
                            "data": {"proxy_list": [f"10.0.0.{index}:8000"]},
                        }
                    )
                if url.endswith("/dynamic/remain"):
                    proxy = str((params or {}).get("proxy", ""))
                    return StubResponse({"code": 200, "data": {proxy: 28}})
                raise AssertionError(f"Unexpected URL: {url}")

        clock = {"value": 1000.0}
        session = StubSession()
        manager = JuliangProxyManager(
            JuliangProxyConfig(
                enabled=True,
                trade_no="1765244755300652",
                api_key="demo-key",
                username="proxy-user",
                password="proxy-pass",
                api_base="http://v2.api.juliangip.com",
                proxy_type=1,
                lease_refresh_margin_seconds=5,
                default_lease_seconds=30,
            ),
            use_env_proxy=False,
            session=session,
            time_fn=lambda: clock["value"],
        )

        first = manager.borrow()
        second = manager.borrow()
        clock["value"] = 1024.0
        third = manager.borrow()

        self.assertEqual(first.endpoint, "10.0.0.1:8000")
        self.assertEqual(second.endpoint, "10.0.0.1:8000")
        self.assertEqual(third.endpoint, "10.0.0.2:8000")
        self.assertEqual(first.username, "proxy-user")
        self.assertEqual(first.password, "proxy-pass")
        getips_calls = [url for url, _ in session.calls if url.endswith("/dynamic/getips")]
        getips_params = [params for url, params in session.calls if url.endswith("/dynamic/getips")]
        remain_calls = [url for url, _ in session.calls if url.endswith("/dynamic/remain")]
        self.assertEqual(len(getips_calls), 2)
        self.assertTrue(all((params or {}).get("auto_white") == 1 for params in getips_params))
        self.assertEqual(len(remain_calls), 2)

    def test_http_client_can_disable_environment_proxy_inheritance(self) -> None:
        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False)
        self.assertFalse(client._session.trust_env)

    def test_http_client_scales_connection_pool_to_worker_count(self) -> None:
        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False, pool_maxsize=16)
        https_adapter = client._session.adapters["https://"]
        self.assertEqual(getattr(https_adapter, "_pool_maxsize", None), 16)

    def test_resolve_worker_count_prefers_parallel_auto_mode_when_proxy_is_enabled(self) -> None:
        self.assertEqual(
            resolve_worker_count(
                requested_workers=0,
                host_max_workers=8,
                effective_symbols=100,
                proxy_enabled=True,
            ),
            8,
        )

    def test_resolve_worker_count_keeps_direct_mode_conservative_by_default(self) -> None:
        self.assertEqual(
            resolve_worker_count(
                requested_workers=0,
                host_max_workers=8,
                effective_symbols=100,
                proxy_enabled=False,
            ),
            1,
        )

    def test_default_end_date_uses_previous_session_before_cutoff(self) -> None:
        class StubCalendar:
            @staticmethod
            def latest_session(upper: dt.date | None = None) -> dt.date:
                if upper == dt.date(2026, 3, 25):
                    return dt.date(2026, 3, 25)
                if upper == dt.date(2026, 3, 24):
                    return dt.date(2026, 3, 24)
                raise AssertionError(f"Unexpected upper={upper}")

        resolved = resolve_default_end_date(
            StubCalendar(),
            now=dt.datetime(2026, 3, 25, 1, 56, 0),
            data_ready_hour=16,
        )

        self.assertEqual(resolved, dt.date(2026, 3, 24))

    def test_default_end_date_keeps_same_day_after_cutoff(self) -> None:
        class StubCalendar:
            @staticmethod
            def latest_session(upper: dt.date | None = None) -> dt.date:
                if upper == dt.date(2026, 3, 25):
                    return dt.date(2026, 3, 25)
                raise AssertionError(f"Unexpected upper={upper}")

        resolved = resolve_default_end_date(
            StubCalendar(),
            now=dt.datetime(2026, 3, 25, 16, 1, 0),
            data_ready_hour=16,
        )

        self.assertEqual(resolved, dt.date(2026, 3, 25))

    def test_parse_dates_without_explicit_end_date_uses_cutoff_aware_default(self) -> None:
        args = SimpleNamespace(start_date="1990-01-01", end_date=None)

        class StubCalendar:
            @staticmethod
            def latest_session(upper: dt.date | None = None) -> dt.date:
                return upper

        with patch("download_stock.resolve_default_end_date", return_value=dt.date(2026, 3, 24)) as default_end_mock:
            start, end = parse_dates(args, calendar=StubCalendar())

        self.assertEqual(start, dt.date(1990, 1, 1))
        self.assertEqual(end, dt.date(2026, 3, 24))
        default_end_mock.assert_called_once()

    def test_existing_but_stale_file_is_not_skipped(self) -> None:
        class StubCalendar:
            @staticmethod
            def next_session(value: dt.date) -> dt.date:
                if value == dt.date(2024, 1, 2):
                    return dt.date(2024, 1, 3)
                raise AssertionError(f"Unexpected date: {value}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "daily"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "open": [1.0], "close": [1.0]})
            frame.to_parquet(parquet_dir / "sz000001.parquet", index=False)

            config = CrawlerConfig(
                start_date=dt.date(2024, 1, 1),
                end_date=dt.date(2024, 1, 10),
                parquet_dir=parquet_dir,
                max_workers=1,
                max_retries=1,
                adjust="hfq",
                force=False,
                timeout=1.0,
                limit=None,
                request_interval=0.0,
                request_jitter=0.0,
                retry_sleep=0.0,
                shuffle_symbols=False,
                report_file=None,
            )
            tracker = OutputTracker(parquet_dir, config.end_date)
            crawler = StockCrawler(
                config,
                ParquetStore(parquet_dir),
                HistoricalQuoteClient(HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0), config),
                [Instrument(code="000001", market="SZ")],
                tracker,
                OutcomeRecorder(None),
                calendar=StubCalendar(),
            )
            pending = list(crawler._pending_symbols())
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].request_start_date, dt.date(2024, 1, 3))

    def test_existing_file_at_required_end_date_is_skipped(self) -> None:
        class StubCalendar:
            @staticmethod
            def next_session(value: dt.date) -> dt.date:
                raise AssertionError(f"Should not request next session for {value}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "daily"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-10"]), "open": [1.0], "close": [1.0]})
            frame.to_parquet(parquet_dir / "sz000001.parquet", index=False)

            config = CrawlerConfig(
                start_date=dt.date(2024, 1, 1),
                end_date=dt.date(2024, 1, 10),
                parquet_dir=parquet_dir,
                max_workers=1,
                max_retries=1,
                adjust="hfq",
                force=False,
                timeout=1.0,
                limit=None,
                request_interval=0.0,
                request_jitter=0.0,
                retry_sleep=0.0,
                shuffle_symbols=False,
                report_file=None,
            )
            tracker = OutputTracker(parquet_dir, config.end_date)
            crawler = StockCrawler(
                config,
                ParquetStore(parquet_dir),
                HistoricalQuoteClient(HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0), config),
                [Instrument(code="000001", market="SZ")],
                tracker,
                OutcomeRecorder(None),
                calendar=StubCalendar(),
            )

            pending = list(crawler._pending_symbols())

            self.assertEqual(pending, [])

    def test_incremental_download_merges_existing_rows_with_new_tail(self) -> None:
        class StubCalendar:
            @staticmethod
            def next_session(value: dt.date) -> dt.date:
                if value == dt.date(2024, 1, 5):
                    return dt.date(2024, 1, 8)
                raise AssertionError(f"Unexpected date: {value}")

        class RecordingQuoteClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dt.date, dt.date]] = []

            def fetch(
                self,
                instrument: Instrument,
                *,
                start_date: dt.date | None = None,
                end_date: dt.date | None = None,
            ) -> FetchResult:
                self.calls.append((instrument.symbol, start_date, end_date))
                frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
                        "open": [3.0, 4.0, 5.0],
                        "close": [3.1, 4.1, 5.1],
                        "symbol": [instrument.symbol] * 3,
                        "factor": [1.0] * 3,
                        "money": [100.0, 100.0, 100.0],
                    }
                )
                return FetchResult(frame=frame, meta=RequestMeta(attempts=1, elapsed_sec=0.1, last_status_code=200))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "daily"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            existing = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                    "open": [1.0, 2.0],
                    "close": [1.1, 2.1],
                    "symbol": ["000001.SZ", "000001.SZ"],
                    "factor": [1.0, 1.0],
                    "money": [100.0, 100.0],
                }
            )
            existing.to_parquet(parquet_dir / "sz000001.parquet", index=False)

            config = CrawlerConfig(
                start_date=dt.date(1990, 1, 1),
                end_date=dt.date(2024, 1, 10),
                parquet_dir=parquet_dir,
                max_workers=1,
                max_retries=1,
                adjust="hfq",
                force=False,
                timeout=1.0,
                limit=None,
                request_interval=0.0,
                request_jitter=0.0,
                retry_sleep=0.0,
                shuffle_symbols=False,
                report_file=root / "report.jsonl",
            )
            tracker = OutputTracker(parquet_dir, config.end_date)
            quote_client = RecordingQuoteClient()
            crawler = StockCrawler(
                config,
                ParquetStore(parquet_dir),
                quote_client,
                [Instrument(code="000001", market="SZ")],
                tracker,
                OutcomeRecorder(config.report_file),
                calendar=StubCalendar(),
            )

            stats = crawler.run()

            self.assertEqual(stats["processed"], 1)
            self.assertEqual(quote_client.calls, [("000001.SZ", dt.date(2024, 1, 8), dt.date(2024, 1, 10))])
            merged = pd.read_parquet(parquet_dir / "sz000001.parquet")
            self.assertEqual(
                [value.date().isoformat() for value in pd.to_datetime(merged["date"]).tolist()],
                ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
            )

    def test_console_progress_bar_renders_single_line_progress(self) -> None:
        stream = io.StringIO()
        progress = ConsoleProgressBar(total=3, stream=stream, enabled=True, interactive=True)

        progress.update(processed=1, failed=0, empty=0, rejected=0, skipped=0)
        progress.close()

        rendered = stream.getvalue()
        self.assertIn("1/3", rendered)
        self.assertIn("ok=1", rendered)

    def test_empty_response_is_not_counted_as_processed(self) -> None:
        class EmptyQuoteClient:
            @staticmethod
            def fetch(
                instrument: Instrument,
                *,
                start_date: dt.date | None = None,
                end_date: dt.date | None = None,
            ) -> FetchResult:
                del instrument, start_date, end_date
                return FetchResult(
                    frame=pd.DataFrame(),
                    meta=RequestMeta(attempts=2, elapsed_sec=0.3, last_status_code=200, anti_bot_suspected=True, detail="blocked"),
                    payload_rc=0,
                    payload_msg="",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "daily"
            config = CrawlerConfig(
                start_date=dt.date(2024, 1, 1),
                end_date=dt.date(2024, 1, 10),
                parquet_dir=parquet_dir,
                max_workers=1,
                max_retries=1,
                adjust="hfq",
                force=False,
                timeout=1.0,
                limit=None,
                request_interval=0.0,
                request_jitter=0.0,
                retry_sleep=0.0,
                shuffle_symbols=False,
                report_file=root / "report.jsonl",
            )
            tracker = OutputTracker(parquet_dir, config.end_date)
            recorder = OutcomeRecorder(config.report_file)
            crawler = StockCrawler(
                config,
                ParquetStore(parquet_dir),
                EmptyQuoteClient(),
                [Instrument(code="000001", market="SZ")],
                tracker,
                recorder,
                calendar=TradingCalendar(),
            )
            stats = crawler.run()
            self.assertEqual(stats["processed"], 0)
            self.assertEqual(stats["rejected"], 1)
            self.assertTrue(config.report_file.exists())

    def test_cookie_warmup_manager_reuses_fresh_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "eastmoney_cookie.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "cookie_header": "nid18=abc; nid18_create_time=123",
                        "fetched_at": "2026-03-22T12:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            config = EastmoneyCookieWarmupConfig(
                enabled=True,
                cache_file=cache_path,
                max_age_seconds=3600,
                node_binary="node",
                script_path=Path("tools/eastmoney_cookie_warmer.mjs"),
                browser_path=None,
                browser_proxy=None,
                timeout_ms=9000,
            )
            manager = EastmoneyCookieWarmupManager(
                config,
                now_fn=lambda: dt.datetime(2026, 3, 22, 12, 30, tzinfo=dt.timezone.utc),
            )

            with patch("download_stock.subprocess.run") as run_mock:
                header = manager.get_cookie_header()

            self.assertEqual(header, "nid18=abc; nid18_create_time=123")
            run_mock.assert_not_called()

    def test_cookie_warmup_manager_refreshes_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = root / "eastmoney_cookie.json"
            script_path = root / "tools" / "eastmoney_cookie_warmer.mjs"
            config = EastmoneyCookieWarmupConfig(
                enabled=True,
                cache_file=cache_path,
                max_age_seconds=3600,
                node_binary="/opt/node/bin/node",
                script_path=script_path,
                browser_path="/usr/bin/chromium",
                browser_proxy="http://127.0.0.1:7890",
                timeout_ms=7000,
            )
            manager = EastmoneyCookieWarmupManager(
                config,
                now_fn=lambda: dt.datetime(2026, 3, 22, 12, 30, tzinfo=dt.timezone.utc),
            )
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "nid18": "new-cookie",
                        "nid18_create_time": "1774187166298",
                        "fetched_at": "2026-03-22T12:30:00+00:00",
                    }
                ),
                stderr="",
            )

            with patch("download_stock.subprocess.run", return_value=completed) as run_mock:
                header = manager.get_cookie_header()

            self.assertEqual(header, "nid18=new-cookie; nid18_create_time=1774187166298")
            self.assertTrue(cache_path.exists())
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cached["cookie_header"], header)
            cmd = run_mock.call_args.args[0]
            self.assertEqual(cmd[0], "/opt/node/bin/node")
            self.assertIn(str(script_path), cmd)
            self.assertIn("--browser-path", cmd)
            self.assertIn("/usr/bin/chromium", cmd)
            self.assertIn("--proxy", cmd)
            self.assertIn("http://127.0.0.1:7890", cmd)
            self.assertIn("--timeout-ms", cmd)
            self.assertIn("7000", cmd)

    def test_cookie_warmup_manager_uses_proxy_state_for_browser_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = EastmoneyCookieWarmupConfig(
                enabled=True,
                cache_file=root / "eastmoney_cookie.json",
                max_age_seconds=0,
                node_binary="node",
                script_path=root / "tools" / "eastmoney_cookie_warmer.mjs",
                browser_path="/usr/bin/chromium",
                browser_proxy=None,
                timeout_ms=8000,
            )
            manager = EastmoneyCookieWarmupManager(config)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "nid18": "new-cookie",
                        "nid18_create_time": "1774187166298",
                        "fetched_at": "2026-03-22T12:30:00+00:00",
                    }
                ),
                stderr="",
            )

            with patch("download_stock.subprocess.run", return_value=completed) as run_mock:
                header = manager.get_cookie_header(
                    proxy_state=ProxyState(endpoint="1.2.3.4:9000", username="proxy-user", password="proxy-pass")
                )

            self.assertEqual(header, "nid18=new-cookie; nid18_create_time=1774187166298")
            cmd = run_mock.call_args.args[0]
            self.assertIn("--proxy", cmd)
            self.assertIn("http://1.2.3.4:9000", cmd)
            self.assertIn("--proxy-username", cmd)
            self.assertIn("proxy-user", cmd)
            self.assertIn("--proxy-password", cmd)
            self.assertIn("proxy-pass", cmd)

    def test_cookie_warmup_manager_returns_empty_string_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EastmoneyCookieWarmupConfig(
                enabled=True,
                cache_file=Path(tmpdir) / "eastmoney_cookie.json",
                max_age_seconds=0,
                node_binary="node",
                script_path=Path("tools/eastmoney_cookie_warmer.mjs"),
                browser_path=None,
                browser_proxy=None,
                timeout_ms=8000,
            )
            manager = EastmoneyCookieWarmupManager(config)

            with patch("download_stock.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["node"])):
                header = manager.get_cookie_header()

            self.assertEqual(header, "")

    def test_http_client_injects_cookie_header_before_request(self) -> None:
        class StubCookieManager:
            def __init__(self) -> None:
                self.calls = 0

            def get_cookie_header(self, force_refresh: bool = False, proxy_state: ProxyState | None = None) -> str:
                del force_refresh
                del proxy_state
                self.calls += 1
                return "nid18=abc; nid18_create_time=123"

        class FakeResponse:
            status_code = 200
            text = '{"data":{"klines":[]}}'
            headers = {"content-type": "application/json; charset=UTF-8"}

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"data": {"klines": []}}

        cookie_manager = StubCookieManager()
        client = HttpClient(
            None,
            1.0,
            1,
            0.0,
            0.0,
            0.0,
            use_env_proxy=False,
            cookie_manager=cookie_manager,
        )

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None) -> FakeResponse:
            del url, params, timeout, proxies
            self.assertEqual(client._session.headers.get("Cookie"), "nid18=abc; nid18_create_time=123")
            return FakeResponse()

        client._session.get = fake_get  # type: ignore[method-assign]
        client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})
        self.assertEqual(cookie_manager.calls, 1)

    def test_http_client_forces_cookie_refresh_even_when_max_retries_is_one(self) -> None:
        class StubCookieManager:
            def __init__(self) -> None:
                self.calls: list[bool] = []

            def get_cookie_header(self, force_refresh: bool = False, proxy_state: ProxyState | None = None) -> str:
                del proxy_state
                self.calls.append(force_refresh)
                if force_refresh:
                    return "nid18=fresh; nid18_create_time=456"
                return "nid18=stale; nid18_create_time=123"

        class FakeResponse:
            status_code = 200
            text = '{"data":{"klines":[]}}'
            headers = {"content-type": "application/json; charset=UTF-8"}

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"data": {"klines": []}}

        cookie_manager = StubCookieManager()
        client = HttpClient(
            None,
            1.0,
            1,
            0.0,
            0.0,
            0.0,
            use_env_proxy=False,
            cookie_manager=cookie_manager,
        )
        attempts = {"count": 0}

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None) -> FakeResponse:
            del url, params, timeout, proxies
            attempts["count"] += 1
            if client._session.headers.get("Cookie") == "nid18=stale; nid18_create_time=123":
                raise requests.exceptions.ConnectionError("stale cookie rejected")
            return FakeResponse()

        client._session.get = fake_get  # type: ignore[method-assign]
        _, meta = client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(meta.attempts, 2)
        self.assertEqual(cookie_manager.calls, [False, True])

    def test_http_client_does_not_carry_previous_attempt_detail_into_next_failure(self) -> None:
        class StubCookieManager:
            def __init__(self) -> None:
                self.calls: list[bool] = []

            def get_cookie_header(self, force_refresh: bool = False, proxy_state: ProxyState | None = None) -> str:
                del proxy_state
                self.calls.append(force_refresh)
                if force_refresh:
                    return "nid18=fresh; nid18_create_time=456"
                return "nid18=stale; nid18_create_time=123"

        cookie_manager = StubCookieManager()
        client = HttpClient(
            None,
            1.0,
            1,
            0.0,
            0.0,
            0.0,
            use_env_proxy=False,
            cookie_manager=cookie_manager,
        )
        attempts = {"count": 0}

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None):
            del url, params, timeout, proxies
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise requests.exceptions.ConnectionError("stale cookie rejected")
            raise requests.exceptions.ConnectionError("fresh cookie still rejected")

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        message = str(ctx.exception)
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(cookie_manager.calls, [False, True, True])
        self.assertEqual(message.count("cookie_refresh=forced"), 0)
        self.assertEqual(message.count("stale cookie rejected"), 0)
        self.assertEqual(message.count("fresh cookie still rejected"), 1)

    def test_http_client_classifies_proxy_transport_failures(self) -> None:
        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False)

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None):
            del url, params, timeout, proxies
            raise requests.exceptions.ProxyError("407 Proxy Authentication Required")

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        self.assertIn("category=proxy_transport", str(ctx.exception))

    def test_http_client_classifies_target_site_failures(self) -> None:
        class FakeResponse:
            status_code = 403
            text = "Access denied verify"
            headers = {"content-type": "text/html; charset=UTF-8"}

            @staticmethod
            def raise_for_status() -> None:
                raise requests.exceptions.HTTPError("403 Client Error")

        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False)

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None) -> FakeResponse:
            del url, params, timeout, proxies
            return FakeResponse()

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        self.assertIn("category=target_site", str(ctx.exception))

    def test_http_client_classifies_proxy_read_timeout_when_proxy_is_in_use(self) -> None:
        class StubProxyManager:
            def __init__(self) -> None:
                self.failures: list[str] = []

            @staticmethod
            def borrow() -> ProxyState:
                return ProxyState(endpoint="1.2.3.4:9000")

            def report_failure(self, state: ProxyState) -> None:
                self.failures.append(state.endpoint)

            @staticmethod
            def report_success(state: ProxyState) -> None:
                del state

        proxy_manager = StubProxyManager()
        client = HttpClient(proxy_manager, 1.0, 0, 0.0, 0.0, 0.0, use_env_proxy=False)

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None):
            del url, params, timeout, proxies
            raise requests.exceptions.ReadTimeout("read timed out")

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        self.assertIn("category=proxy_transport", str(ctx.exception))
        self.assertEqual(proxy_manager.failures, ["1.2.3.4:9000"])

    def test_http_client_classifies_chunked_broken_pipe_as_proxy_transport(self) -> None:
        class StubProxyManager:
            def __init__(self) -> None:
                self.failures: list[str] = []

            @staticmethod
            def borrow() -> ProxyState:
                return ProxyState(endpoint="1.2.3.4:9000")

            def report_failure(self, state: ProxyState) -> None:
                self.failures.append(state.endpoint)

            @staticmethod
            def report_success(state: ProxyState) -> None:
                del state

        proxy_manager = StubProxyManager()
        client = HttpClient(proxy_manager, 1.0, 0, 0.0, 0.0, 1.5, use_env_proxy=False)

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None):
            del url, params, timeout, proxies
            raise requests.exceptions.ChunkedEncodingError(
                "(\"Connection broken: BrokenPipeError(32, 'Broken pipe')\", BrokenPipeError(32, 'Broken pipe'))"
            )

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        self.assertIn("category=proxy_transport", str(ctx.exception))
        self.assertEqual(proxy_manager.failures, ["1.2.3.4:9000"])

    def test_http_client_treats_max_retries_as_retry_budget(self) -> None:
        class FakeResponse:
            status_code = 200
            text = '{"data":{"klines":[]}}'
            headers = {"content-type": "application/json; charset=UTF-8"}

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"data": {"klines": []}}

        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False)
        attempts = {"count": 0}

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None) -> FakeResponse:
            del url, params, timeout, proxies
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise requests.exceptions.ConnectionError("first attempt failed")
            return FakeResponse()

        client._session.get = fake_get  # type: ignore[method-assign]
        _, meta = client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(meta.attempts, 2)

    def test_http_client_retries_proxy_transport_without_sleeping(self) -> None:
        class FakeResponse:
            status_code = 200
            text = '{"data":{"klines":[]}}'
            headers = {"content-type": "application/json; charset=UTF-8"}

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"data": {"klines": []}}

        class StubProxyManager:
            @staticmethod
            def borrow() -> ProxyState:
                return ProxyState(endpoint="1.2.3.4:9000")

            @staticmethod
            def report_failure(state: ProxyState) -> None:
                del state

            @staticmethod
            def report_success(state: ProxyState) -> None:
                del state

        client = HttpClient(StubProxyManager(), 1.0, 1, 0.0, 0.0, 1.5, use_env_proxy=False)
        attempts = {"count": 0}

        def fake_get(url: str, params: dict, timeout: float, proxies: dict | None):
            del url, params, timeout, proxies
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise requests.exceptions.ProxyError("proxy transport failed")
            return FakeResponse()

        client._session.get = fake_get  # type: ignore[method-assign]

        with patch("download_stock.time.sleep") as sleep_mock:
            _, meta = client.get_json(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                {"secid": "1.600000"},
            )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(meta.attempts, 2)
        sleep_mock.assert_not_called()

    def test_stock_crawler_records_request_failure_category(self) -> None:
        class ExplodingQuoteClient:
            @staticmethod
            def fetch(
                instrument: Instrument,
                *,
                start_date: dt.date | None = None,
                end_date: dt.date | None = None,
            ) -> FetchResult:
                del instrument, start_date, end_date
                raise RequestFailedError(
                    "proxy failed",
                    RequestMeta(
                        attempts=3,
                        elapsed_sec=1.2,
                        last_status_code=407,
                        anti_bot_suspected=False,
                        proxy_endpoint="1.2.3.4:8000",
                        failure_category="proxy_transport",
                        detail="category=proxy_transport",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "daily"
            report_file = root / "report.jsonl"
            config = CrawlerConfig(
                start_date=dt.date(2024, 1, 1),
                end_date=dt.date(2024, 1, 10),
                parquet_dir=parquet_dir,
                max_workers=1,
                max_retries=1,
                adjust="hfq",
                force=False,
                timeout=1.0,
                limit=None,
                request_interval=0.0,
                request_jitter=0.0,
                retry_sleep=0.0,
                shuffle_symbols=False,
                report_file=report_file,
            )
            crawler = StockCrawler(
                config,
                ParquetStore(parquet_dir),
                ExplodingQuoteClient(),
                [Instrument(code="000001", market="SZ")],
                OutputTracker(parquet_dir, config.end_date),
                OutcomeRecorder(report_file),
                calendar=TradingCalendar(),
            )

            stats = crawler.run()

            self.assertEqual(stats["failed"], 1)
            payload = json.loads(report_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["failure_category"], "proxy_transport")
            self.assertEqual(payload["last_status_code"], 407)


if __name__ == "__main__":
    unittest.main()
