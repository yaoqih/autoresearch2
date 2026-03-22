from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from download_stock import (
    CrawlerConfig,
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


if __name__ == "__main__":
    unittest.main()
