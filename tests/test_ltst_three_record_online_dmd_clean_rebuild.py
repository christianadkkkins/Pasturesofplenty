import unittest

import numpy as np
import pandas as pd

from artifact.ltst_three_record_online_dmd_clean_rebuild import (
    CleanDMDConfig,
    build_pressure_score,
    extract_dmd_frame,
    feature_is_rankable,
    stabilize_modal_frame,
    summarize_event_windows,
)


class LTSTThreeRecordOnlineDMDCleanRebuildTest(unittest.TestCase):
    def test_feature_is_rankable_blocks_labels(self) -> None:
        self.assertFalse(feature_is_rankable("future_onset_within_30"))
        self.assertFalse(feature_is_rankable("event"))
        self.assertFalse(feature_is_rankable("event_onset"))
        self.assertFalse(feature_is_rankable("at_risk"))
        self.assertTrue(feature_is_rankable("dominant_radius"))
        self.assertTrue(feature_is_rankable("dmd_observer_proj_volume_xsl"))

    def test_stabilize_modal_frame_clips_and_finite(self) -> None:
        frame = pd.DataFrame(
            {
                "dominant_radius": [0.0, 1.0, 2.0, 1e12, -1e12],
                "lambda_0_abs": [1.0, 1.1, np.nan, np.inf, -np.inf],
            }
        )
        stabilized, cols = stabilize_modal_frame(frame, ["dominant_radius", "lambda_0_abs"], clip=6.0)
        self.assertEqual(cols, ["stable_dominant_radius", "stable_lambda_0_abs"])
        self.assertTrue(np.isfinite(stabilized[cols].to_numpy(dtype=float)).all())
        self.assertLessEqual(np.nanmax(np.abs(stabilized[cols].to_numpy(dtype=float))), 6.0)

    def test_pressure_score_builds_on_ranked_features(self) -> None:
        cfg = CleanDMDConfig()
        rows = 520
        time = np.arange(rows) - 400
        df = pd.DataFrame(
            {
                "time_seconds": time.astype(float),
                "event_onset": 0,
                "event": 0,
                "dominant_radius": np.r_[np.zeros(400), np.ones(120)],
                "dmd_observer_proj_volume_xsl": np.r_[np.zeros(400), np.linspace(0, 2, 120)],
            }
        )
        df.loc[400, "event_onset"] = 1
        rank, _, bg_df = summarize_event_windows(df, ["dominant_radius", "dmd_observer_proj_volume_xsl"], cfg)
        scored, feats = build_pressure_score(df, rank, bg_df, topn=1, score_name="score")
        self.assertEqual(len(feats), 1)
        self.assertIn("score", scored.columns)

    def test_extract_dmd_frame_is_shifted_causally(self) -> None:
        cfg = CleanDMDConfig(dmd_init_batch=2, top_k_eigs=1)
        g = pd.DataFrame(
            {
                "time_seconds": [0.0, 1.0, 2.0, 3.0, 4.0],
                "event": [0, 0, 0, 1, 1],
                "event_onset": [0, 0, 1, 0, 0],
                "a": [0.0, 0.1, 0.2, 0.3, 0.4],
                "b": [0.0, -0.1, -0.2, -0.3, -0.4],
            }
        )
        rows = extract_dmd_frame([("sX", g, ["a", "b"])], cfg)
        self.assertFalse(rows.empty)
        self.assertGreaterEqual(rows["row_idx"].min(), cfg.dmd_init_batch + 1)
        self.assertIn(3.0, rows["time_seconds"].to_list())


if __name__ == "__main__":
    unittest.main()
