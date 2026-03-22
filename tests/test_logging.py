from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_impl.cli import build_parser
from quant_impl.pipelines.daily import run_download_step
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
            expected_log = str((config.paths.logs_dir / "download.log").resolve())
            expected_report = str((root / "artifacts" / "logs" / "download_report.jsonl").resolve())
            self.assertIn(expected_log, cmd)
            self.assertIn(expected_report, cmd)
            self.assertEqual(result["log_file"], expected_log)
            self.assertEqual(result["report_file"], expected_report)

    def test_parser_accepts_logging_overrides_after_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["download", "--log-level", "DEBUG", "--log-file", "logs/custom.log"])
        self.assertEqual(args.command, "download")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.log_file, "logs/custom.log")


if __name__ == "__main__":
    unittest.main()
