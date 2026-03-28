from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from quant_impl.data.market import make_day_batches, transform_training_targets
from quant_impl.modeling.ranker import blocked_top_fillable_pairwise_loss
from quant_impl.pipelines.train import _build_training_summary, train_pipeline
from quant_impl.settings import AppConfig

from tests.helpers import make_synthetic_market_parquet, make_test_config


class ChampionSpecTest(unittest.TestCase):
    def test_exec_fillable_rank_neg1_transform_pushes_blocked_names_to_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(Path(tmpdir))
            config.training.target_transform = "exec_fillable_rank_neg1"
            targets = torch.tensor([0.08, 0.01, -0.05], dtype=torch.float32)
            blocked = torch.tensor([0, 1, 0], dtype=torch.uint8)
            transformed = transform_training_targets(targets, [3], config.training, blocked_flags=blocked)
            expected = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
            self.assertTrue(torch.allclose(transformed, expected))

    def test_make_day_batches_applies_train_only_abs_cap(self) -> None:
        bundle = {
            "features": torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [5.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "targets": torch.tensor([0.02, 0.15, -0.05, 0.11, -0.03], dtype=torch.float32),
            "asset_ids": torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32),
            "day_ptr": torch.tensor([0, 3, 5], dtype=torch.int64),
            "dates": ["2020-01-01", "2020-01-02"],
        }
        batch = next(
            make_day_batches(
                bundle,
                day_indices=[0, 1],
                batch_days=2,
                clip_value=5.0,
                target_abs_cap=0.10,
            )
        )
        self.assertEqual(batch["group_sizes"], [2, 1])
        self.assertEqual(batch["day_indices"], [0, 1])
        self.assertTrue(bool((batch["targets"].abs() <= 0.10).all().item()))

    def test_blocked_pairwise_auxiliary_penalizes_blocked_names_outranking_fillable_names(self) -> None:
        scores = torch.tensor([0.9, 0.2, 0.8, 0.1], dtype=torch.float32)
        raw_targets = torch.tensor([0.01, 0.08, 0.06, -0.02], dtype=torch.float32)
        blocked = torch.tensor([1, 0, 0, 0], dtype=torch.uint8)
        loss = blocked_top_fillable_pairwise_loss(
            scores,
            raw_targets,
            [4],
            blocked,
            top_fraction=0.10,
        )
        self.assertGreater(float(loss.item()), 0.0)

    def test_training_summary_matches_reference_style_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(Path(tmpdir))
            fold_summaries = [
                {
                    "valid": {
                        "daily": {
                            "dates": ["2020-01-01", "2020-01-02"],
                            "selected_returns": [0.10, -0.20],
                            "universe_returns": [0.0, 0.0],
                            "oracle_returns": [0.10, 0.10],
                        }
                    },
                    "holdout": {
                        "daily": {
                            "dates": ["2020-02-01", "2020-02-02"],
                            "selected_returns": [0.10, -0.30],
                            "universe_returns": [0.0, 0.0],
                            "oracle_returns": [0.10, 0.10],
                        }
                    },
                },
                {
                    "valid": {
                        "daily": {
                            "dates": ["2020-03-01", "2020-03-02"],
                            "selected_returns": [0.05, 0.05],
                            "universe_returns": [0.0, 0.0],
                            "oracle_returns": [0.10, 0.10],
                        }
                    },
                    "holdout": {
                        "daily": {
                            "dates": ["2020-04-01", "2020-04-02"],
                            "selected_returns": [0.20, 0.20],
                            "universe_returns": [0.0, 0.0],
                            "oracle_returns": [0.10, 0.10],
                        }
                    },
                },
            ]
            config.training.recent_holdout_folds = 1
            summary = _build_training_summary(config, fold_summaries)
            self.assertAlmostEqual(summary["cv_holdout_mean_return"], 0.05)
            self.assertLess(summary["cv_holdout_max_drawdown"], -0.25)
            self.assertGreater(summary["recent_holdout_mean_return"], summary["cv_holdout_mean_return"])
            self.assertIn("cv_holdout_ret_dd", summary)
            self.assertIn("cv_holdout_trade_rate", summary)
            self.assertIn("cv_holdout_block_rate_open_limit", summary)
            self.assertIn("cv_holdout_switch_rate", summary)
            self.assertIn("cv_holdout_all_fallback_blocked_rate", summary)
            self.assertIn("cv_holdout_one_word_mean_return", summary)

    def test_default_hyperparameters_match_champion_reference(self) -> None:
        config = AppConfig()
        self.assertEqual(config.model.num_residual_blocks, 2)
        self.assertEqual(config.model.shortlist_size, 96)
        self.assertAlmostEqual(config.model.rerank_mix, 0.25)
        self.assertEqual(config.training.pair_samples_per_day, 96)
        self.assertAlmostEqual(config.training.pair_focus_fraction, 0.33)
        self.assertAlmostEqual(config.training.top_bucket_expansion_scale, 10.0)
        self.assertAlmostEqual(config.training.listwise_loss_weight, 0.51)
        self.assertAlmostEqual(config.training.rerank_listwise_loss_weight, 0.06)
        self.assertEqual(config.training.execution_aux_mode, "blocked_pairwise")
        self.assertAlmostEqual(config.training.execution_aux_weight, 0.10)
        self.assertAlmostEqual(config.training.execution_aux_top_fraction, 0.10)
        self.assertIn("blockpair_w0p10", config.data.cache_version)
        self.assertIn("blockpair_w0p10", config.inference.prediction_name)

    def test_train_pipeline_persists_champion_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            result = train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)
            self.assertEqual(result["champion_spec"]["member_config"], "lc96")
            self.assertEqual(result["champion_spec"]["target_transform"], "exec_fillable_rank_neg1")
            self.assertEqual(result["champion_spec"]["execution_aux_mode"], "blocked_pairwise")
            self.assertAlmostEqual(result["champion_spec"]["execution_aux_weight"], 0.10)
            self.assertAlmostEqual(result["champion_spec"]["train_target_abs_cap"], 0.10)
            self.assertTrue(artifact["champion_spec"]["train_target_cap_applies_to_linear_head"])
            self.assertTrue(artifact["champion_spec"]["strict_executable_eval"])
            self.assertEqual(artifact["champion_spec"]["execution_block_rule"], "t+1_hybrid_limit=>rollover_top3_else_cash")
            self.assertEqual(artifact["champion_spec"]["execution_fallback_top_k"], 3)
            self.assertEqual(artifact["champion_spec"]["execution_block_mode"], "hybrid")


if __name__ == "__main__":
    unittest.main()
