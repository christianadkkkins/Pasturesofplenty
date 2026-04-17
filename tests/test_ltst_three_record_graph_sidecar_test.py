from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact.ltst_three_record_graph_sidecar_test import (
    SidecarTestConfig,
    build_pressure_score,
    compute_updated_observer_features,
    summarize_event_windows,
)


class LTSTThreeRecordGraphSidecarTests(unittest.TestCase):
    def test_updated_observer_features_expose_new_geometry_columns(self) -> None:
        n_rows = 24
        t = np.linspace(0.1, 2.2, n_rows, dtype=float)
        x_state = np.stack(
            [
                np.sin(1.2 * t),
                np.cos(0.7 * t + 0.1),
                np.sin(0.5 * t + 0.3) + 0.2 * t,
                np.cos(1.7 * t) - 0.1 * t,
                np.sin(0.9 * t - 0.2),
                np.cos(1.3 * t + 0.5),
            ],
            axis=1,
        )

        observer = compute_updated_observer_features(x_state, SidecarTestConfig())

        expected = {
            "observer_pair_angle_sl",
            "observer_oriented_volume_xsl",
            "observer_sheaf_defect_log_ratio",
            "observer_lie_transition_score",
            "observer_combined_transition_score",
            "observer_A_l",
            "observer_phi_second_order_gated",
            "observer_omega_sl",
        }
        self.assertTrue(expected.issubset(set(observer.columns)))
        for column in expected:
            values = observer[column].to_numpy(dtype=float)
            self.assertEqual(values.shape, (n_rows,))
            self.assertTrue(np.isfinite(np.nan_to_num(values, nan=0.0)).all())

    def test_window_ranking_and_pressure_score_stay_non_null(self) -> None:
        n_rows = 420
        time_seconds = np.arange(-300, 120, dtype=float)
        event_onset = np.zeros(n_rows, dtype=int)
        event_onset[300] = 1
        event_active = (time_seconds >= 0).astype(int)

        panel = pd.DataFrame(
            {
                "time_seconds": time_seconds,
                "event_onset": event_onset,
                "event_active": event_active,
                "event": event_active,
                "graph_a": np.sin(np.linspace(0, 10, n_rows)),
                "graph_b": np.cos(np.linspace(0, 8, n_rows)),
                "observer_pair_angle_sl": np.r_[np.zeros(200), np.linspace(0, 2.5, 100), np.full(120, 2.5)],
                "observer_sheaf_defect_log_ratio": np.r_[np.zeros(210), np.linspace(0, 2.0, 90), np.full(120, 2.0)],
            }
        )

        cfg = SidecarTestConfig(bg_rows=120, gap_rows=20, pre_rows=80, post_rows=40)
        rank, _, bg_df, _, _ = summarize_event_windows(
            panel,
            ["observer_pair_angle_sl", "observer_sheaf_defect_log_ratio"],
            cfg,
        )
        scored, feats = build_pressure_score(panel, rank, bg_df, topn=2)

        self.assertEqual(len(feats), 2)
        self.assertTrue("event_pressure_score" in scored.columns)
        self.assertFalse(scored["event_pressure_score"].isna().any())
        self.assertGreater(
            float(scored.loc[250:299, "event_pressure_score"].mean()),
            float(scored.loc[40:120, "event_pressure_score"].mean()),
        )


if __name__ == "__main__":
    unittest.main()
