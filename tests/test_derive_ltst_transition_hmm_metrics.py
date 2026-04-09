from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from results.derive_ltst_transition_hmm_metrics import compute_metrics


class DeriveLtstTransitionHmmMetricsTests(unittest.TestCase):
    def test_compute_metrics_on_small_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            with (run_dir / "ltst_transition_hmm_record_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "record",
                        "regime",
                        "phenotype_target",
                        "n_beats",
                        "n_st_episodes",
                    ]
                )
                writer.writerow(["rec1", "long_only", "constrained_orbit", 10, 1])

            with (run_dir / "ltst_transition_hmm_event_table.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "record",
                        "regime",
                        "phenotype_target",
                        "st_episode_id",
                        "event_start_idx",
                        "event_end_idx",
                    ]
                )
                writer.writerow(["rec1", "long_only", "constrained_orbit", 0, 5, 6])

            with (run_dir / "ltst_transition_hmm_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "episode_id",
                        "start_idx",
                        "end_idx",
                        "record",
                        "regime",
                        "phenotype_target",
                        "method",
                    ]
                )
                writer.writerow([0, 4, 5, "rec1", "long_only", "constrained_orbit", "hmm"])
                writer.writerow([1, 8, 8, "rec1", "long_only", "constrained_orbit", "hmm"])

            metrics = dict(compute_metrics(run_dir=run_dir, pre_event_beats=2))

            self.assertEqual(metrics["n_events"], 1)
            self.assertEqual(metrics["total_beats"], 10)
            self.assertEqual(metrics["predicted_positive_beats"], 3)
            self.assertEqual(metrics["lead_positive_beats"], 4)
            self.assertEqual(metrics["strict_positive_beats"], 2)
            self.assertEqual(metrics["lead_true_positive_beats"], 2)
            self.assertEqual(metrics["lead_false_positive_beats"], 1)
            self.assertEqual(metrics["lead_true_negative_beats"], 5)
            self.assertEqual(metrics["lead_false_negative_beats"], 2)
            self.assertEqual(metrics["strict_true_positive_beats"], 1)
            self.assertEqual(metrics["strict_false_positive_beats"], 2)
            self.assertEqual(metrics["strict_true_negative_beats"], 6)
            self.assertEqual(metrics["strict_false_negative_beats"], 1)
            self.assertAlmostEqual(metrics["lead_specificity"], 5 / 6)
            self.assertAlmostEqual(metrics["lead_sensitivity"], 0.5)
            self.assertAlmostEqual(metrics["lead_precision"], 2 / 3)
            self.assertAlmostEqual(metrics["strict_specificity"], 0.75)
            self.assertAlmostEqual(metrics["strict_sensitivity"], 0.5)
            self.assertAlmostEqual(metrics["strict_precision"], 1 / 3)
            self.assertAlmostEqual(metrics["alert_occupancy_fraction"], 0.3)
            self.assertEqual(metrics["episode_count"], 2)
            self.assertEqual(metrics["false_positive_episode_count"], 1)
            self.assertAlmostEqual(metrics["false_positive_episode_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
