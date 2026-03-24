from __future__ import annotations

import datetime as dt
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from download_stock import (
    CrawlerConfig,
    EastmoneyCookieWarmupConfig,
    EastmoneyCookieWarmupManager,
    FetchResult,
    HistoricalQuoteClient,
    HttpClient,
    Instrument,
    OutcomeRecorder,
    OutputTracker,
    ParquetStore,
    RequestMeta,
    StockCrawler,
)


class DownloaderPendingTest(unittest.TestCase):
    def test_http_client_can_disable_environment_proxy_inheritance(self) -> None:
        client = HttpClient(None, 1.0, 1, 0.0, 0.0, 0.0, use_env_proxy=False)
        self.assertFalse(client._session.trust_env)

    def test_existing_but_stale_file_is_not_skipped(self) -> None:
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
            )
            pending = list(crawler._pending_symbols())
            self.assertEqual(len(pending), 1)

    def test_empty_response_is_not_counted_as_processed(self) -> None:
        class EmptyQuoteClient:
            @staticmethod
            def fetch(instrument: Instrument) -> FetchResult:
                del instrument
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

    def test_http_client_injects_cookie_header_before_request(self) -> None:
        class StubCookieManager:
            def __init__(self) -> None:
                self.calls = 0

            def get_cookie_header(self, force_refresh: bool = False) -> str:
                del force_refresh
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

            def get_cookie_header(self, force_refresh: bool = False) -> str:
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
                raise requests.ConnectionError("stale cookie rejected")
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

            def get_cookie_header(self, force_refresh: bool = False) -> str:
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
                raise requests.ConnectionError("stale cookie rejected")
            raise requests.ConnectionError("fresh cookie still rejected")

        client._session.get = fake_get  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            client.get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": "1.600000"})

        message = str(ctx.exception)
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(cookie_manager.calls, [False, True])
        self.assertEqual(message.count("cookie_refresh=forced"), 0)
        self.assertEqual(message.count("stale cookie rejected"), 0)
        self.assertEqual(message.count("fresh cookie still rejected"), 1)


if __name__ == "__main__":
    unittest.main()
