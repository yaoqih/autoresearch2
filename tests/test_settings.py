from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_impl.settings import load_config


class SettingsTest(unittest.TestCase):
    def test_load_config_reads_download_use_env_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        "  raw_daily_dir: data/daily",
                        "download:",
                        "  use_env_proxy: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertFalse(config.download.use_env_proxy)

    def test_load_config_reads_eastmoney_cookie_warmer_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        "  raw_daily_dir: data/daily",
                        "download:",
                        "  eastmoney_cookie_warmup: true",
                        "  eastmoney_cookie_cache_file: artifacts/cache/eastmoney_cookie.json",
                        "  eastmoney_cookie_max_age_seconds: 7200",
                        "  eastmoney_cookie_node_binary: /opt/node/bin/node",
                        "  eastmoney_cookie_script: tools/eastmoney_cookie_warmer.mjs",
                        "  eastmoney_browser_path: /usr/bin/chromium",
                        "  eastmoney_browser_proxy: http://127.0.0.1:7890",
                        "  eastmoney_cookie_timeout_ms: 9000",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.download.eastmoney_cookie_warmup)
            self.assertEqual(config.download.eastmoney_cookie_cache_file, "artifacts/cache/eastmoney_cookie.json")
            self.assertEqual(config.download.eastmoney_cookie_max_age_seconds, 7200)
            self.assertEqual(config.download.eastmoney_cookie_node_binary, "/opt/node/bin/node")
            self.assertEqual(config.download.eastmoney_cookie_script, "tools/eastmoney_cookie_warmer.mjs")
            self.assertEqual(config.download.eastmoney_browser_path, "/usr/bin/chromium")
            self.assertEqual(config.download.eastmoney_browser_proxy, "http://127.0.0.1:7890")
            self.assertEqual(config.download.eastmoney_cookie_timeout_ms, 9000)

    def test_load_config_reads_juliang_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        "  raw_daily_dir: data/daily",
                        "download:",
                        "  juliang_enabled: true",
                        "  juliang_api_base: http://v2.api.juliangip.com",
                        "  juliang_proxy_type: 2",
                        "  juliang_lease_refresh_margin_seconds: 7",
                        "  juliang_default_lease_seconds: 35",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.download.juliang_enabled)
            self.assertEqual(config.download.juliang_api_base, "http://v2.api.juliangip.com")
            self.assertEqual(config.download.juliang_proxy_type, 2)
            self.assertEqual(config.download.juliang_lease_refresh_margin_seconds, 7)
            self.assertEqual(config.download.juliang_default_lease_seconds, 35)

    def test_load_config_defaults_cookie_warmup_off(self) -> None:
        config = load_config()
        self.assertFalse(config.download.eastmoney_cookie_warmup)
        self.assertEqual(config.download.max_workers, 0)
        self.assertEqual(config.download.host_max_workers, 4)
        self.assertEqual(config.download.max_retries, 2)
        self.assertEqual(config.download.request_interval, 0.0)
        self.assertEqual(config.download.request_jitter, 0.0)
        self.assertEqual(config.inference.execution_fallback_top_k, 5)

    def test_load_config_reads_deployment_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        "  raw_daily_dir: data/daily",
                        "training:",
                        "  deployment_epochs: 7",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(getattr(config.training, "deployment_epochs", None), 7)

    def test_prediction_archive_paths(self) -> None:
        config = load_config()

        self.assertEqual(config.prediction_daily_dir, config.paths.predictions_dir / "daily")
        self.assertEqual(
            config.prediction_daily_path("2026-03-24"),
            config.paths.predictions_dir / "daily" / "2026-03-24.json",
        )
        self.assertEqual(config.prediction_index_path, config.paths.predictions_dir / "index.json")
        self.assertEqual(config.prediction_latest_path, config.paths.predictions_dir / "latest.json")


if __name__ == "__main__":
    unittest.main()
