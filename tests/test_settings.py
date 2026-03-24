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


if __name__ == "__main__":
    unittest.main()
