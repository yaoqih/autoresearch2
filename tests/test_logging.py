from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_impl.cli import build_parser
from quant_impl.pipelines.daily import run_download_step
from quant_impl.settings import REPO_ROOT, load_config
from quant_impl.utils.logging_utils import setup_logging

from tests.helpers import make_test_config


class LoggingIntegrationTest(unittest.TestCase):
    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    def test_setup_logging_writes_default_command_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            log_path = setup_logging(config, "train")
            logging.getLogger("quant_impl.test").info("training log smoke")
            for handler in logging.getLogger().handlers:
                handler.flush()
            self.assertEqual(log_path, (config.paths.logs_dir / "train.log").resolve())
            self.assertTrue(log_path.exists())
            self.assertIn("training log smoke", log_path.read_text(encoding="utf-8"))

    def test_run_download_step_uses_global_logging_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.logging.level = "DEBUG"
            config.logging.console = False
            config.download.use_env_proxy = False
            config.download.report_file = str(root / "artifacts" / "logs" / "download_report.jsonl")
            with patch("quant_impl.pipelines.daily.subprocess.run") as run_mock:
                run_mock.return_value = object()
                result = run_download_step(config)
            cmd = run_mock.call_args.args[0]
            kwargs = run_mock.call_args.kwargs
            self.assertTrue(kwargs["check"])
            self.assertNotIn("capture_output", kwargs)
            self.assertIn("--log-level", cmd)
            self.assertIn("DEBUG", cmd)
            self.assertIn("--no-console-log", cmd)
            self.assertIn("--no-use-env-proxy", cmd)
            expected_log = str((config.paths.logs_dir / "download.log").resolve())
            expected_report = str((root / "artifacts" / "logs" / "download_report.jsonl").resolve())
            self.assertIn(expected_log, cmd)
            self.assertIn(expected_report, cmd)
            self.assertEqual(result["log_file"], expected_log)
            self.assertEqual(result["report_file"], expected_report)

    def test_run_download_step_passes_eastmoney_cookie_warmer_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.download.eastmoney_cookie_warmup = True
            config.download.eastmoney_cookie_cache_file = str(root / "artifacts" / "cache" / "eastmoney_cookie.json")
            config.download.eastmoney_cookie_max_age_seconds = 1800
            config.download.eastmoney_cookie_node_binary = "/opt/node/bin/node"
            config.download.eastmoney_cookie_script = "tools/eastmoney_cookie_warmer.mjs"
            config.download.eastmoney_browser_path = "/usr/bin/chromium"
            config.download.eastmoney_browser_proxy = "http://127.0.0.1:7890"
            config.download.eastmoney_cookie_timeout_ms = 7000

            with patch("quant_impl.pipelines.daily.subprocess.run") as run_mock:
                run_mock.return_value = object()
                run_download_step(config)

            cmd = run_mock.call_args.args[0]
            self.assertIn("--eastmoney-cookie-warmup", cmd)
            self.assertIn("--eastmoney-cookie-cache-file", cmd)
            self.assertIn(str((root / "artifacts" / "cache" / "eastmoney_cookie.json").resolve()), cmd)
            self.assertIn("--eastmoney-cookie-max-age-seconds", cmd)
            self.assertIn("1800", cmd)
            self.assertIn("--eastmoney-cookie-node-binary", cmd)
            self.assertIn("/opt/node/bin/node", cmd)
            self.assertIn("--eastmoney-cookie-script", cmd)
            self.assertIn(str((REPO_ROOT / "tools" / "eastmoney_cookie_warmer.mjs").resolve()), cmd)
            self.assertIn("--eastmoney-browser-path", cmd)
            self.assertIn("/usr/bin/chromium", cmd)
            self.assertIn("--eastmoney-browser-proxy", cmd)
            self.assertIn("http://127.0.0.1:7890", cmd)
            self.assertIn("--eastmoney-cookie-timeout-ms", cmd)
            self.assertIn("7000", cmd)

    def test_run_download_step_passes_juliang_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.download.juliang_enabled = True
            config.download.juliang_api_base = "http://v2.api.juliangip.com"
            config.download.juliang_proxy_type = 2
            config.download.juliang_lease_refresh_margin_seconds = 9
            config.download.juliang_default_lease_seconds = 31

            with patch("quant_impl.pipelines.daily.subprocess.run") as run_mock:
                run_mock.return_value = object()
                run_download_step(config)

            cmd = run_mock.call_args.args[0]
            self.assertIn("--juliang-enabled", cmd)
            self.assertIn("--juliang-api-base", cmd)
            self.assertIn("http://v2.api.juliangip.com", cmd)
            self.assertIn("--juliang-proxy-type", cmd)
            self.assertIn("2", cmd)
            self.assertIn("--juliang-lease-refresh-margin-seconds", cmd)
            self.assertIn("9", cmd)
            self.assertIn("--juliang-default-lease-seconds", cmd)
            self.assertIn("31", cmd)

    def test_load_config_reads_default_use_env_proxy_setting(self) -> None:
        config = load_config()
        self.assertFalse(config.download.use_env_proxy)
        self.assertFalse(config.download.eastmoney_cookie_warmup)

    def test_parser_accepts_logging_overrides_after_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["download", "--log-level", "DEBUG", "--log-file", "logs/custom.log"])
        self.assertEqual(args.command, "download")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.log_file, "logs/custom.log")

    def test_parser_accepts_train_deploy_only_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--deploy-only"])
        self.assertEqual(args.command, "train")
        self.assertTrue(args.deploy_only)


if __name__ == "__main__":
    unittest.main()
