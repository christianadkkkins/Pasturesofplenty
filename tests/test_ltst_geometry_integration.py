from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifact"
for path in (REPO_ROOT, ARTIFACT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_lie import CANONICAL_LIE_FEATURE_COLUMNS, CANONICAL_PROJECTIVE_REFERENCE_COLUMNS
from ltst_full_86_study import LTSTConfig, append_shared_geometry_features
from ltst_transition_hmm import TransitionHMMConfig, build_relative_feature_frame


class LtstGeometryIntegrationTests(unittest.TestCase):
    def _synthetic_x_center(self, n_rows: int, dim: int) -> np.ndarray:
        t = np.linspace(0.0, 2.0 * np.pi, n_rows, dtype=float)
        cols = []
        for idx in range(dim):
            cols.append(np.sin((idx + 1) * t / 3.0) + 0.1 * idx * np.cos(t))
        x = np.stack(cols, axis=1)
        return x - x.mean(axis=1, keepdims=True)

    def test_append_shared_geometry_features_adds_canonical_columns(self) -> None:
        n_rows = 18
        cfg = LTSTConfig()
        df = pd.DataFrame(
            {
                "record": ["rec1"] * n_rows,
                "beat_sample": np.arange(100, 100 + n_rows),
                "beat_symbol": ["N"] * n_rows,
                "rr_samples": np.full(n_rows, 10.0),
                "st_event": [False] * (n_rows - 2) + [True, True],
                "st_episode_id": [-1] * (n_rows - 2) + [0, 0],
            }
        )
        out = append_shared_geometry_features(df, x_center=self._synthetic_x_center(n_rows, cfg.dim), cfg=cfg)

        for column in CANONICAL_PROJECTIVE_REFERENCE_COLUMNS + CANONICAL_LIE_FEATURE_COLUMNS:
            self.assertIn(column, out.columns)
            finite_count = int(np.isfinite(out[column].to_numpy(dtype=float)).sum())
            self.assertGreaterEqual(finite_count, n_rows - 1)

    def test_build_relative_feature_frame_exports_hmm_and_sidecar_scores(self) -> None:
        n_rows = 24
        cfg = LTSTConfig()
        base = pd.DataFrame(
            {
                "record": ["rec1"] * n_rows,
                "regime": ["long_only"] * n_rows,
                "phenotype_target": ["constrained_orbit"] * n_rows,
                "beat_sample": np.arange(1000, 1000 + n_rows),
                "beat_symbol": ["N"] * n_rows,
                "rr_samples": np.full(n_rows, 8.0),
                "st_event": [False] * 20 + [True] * 4,
                "st_episode_id": [-1] * 20 + [0] * 4,
                "energy_asym": np.linspace(-0.2, 0.2, n_rows),
                "drift_norm": np.linspace(0.05, 0.25, n_rows),
                "poincare_b": np.linspace(0.7, 1.1, n_rows),
                "kernel_score_angle": np.linspace(0.1, 0.8, n_rows),
                "kernel_score_long": np.linspace(0.2, 0.9, n_rows),
                "kernel_score_hybrid": np.linspace(0.15, 0.85, n_rows),
            }
        )
        beat_df = append_shared_geometry_features(base, x_center=self._synthetic_x_center(n_rows, cfg.dim), cfg=cfg)
        frame = build_relative_feature_frame(beat_df, TransitionHMMConfig(run_dir=Path(".")))

        expected_columns = [
            "hmm_transition_score",
            "hmm_transition_score_z",
            "projective_level_norm",
            "projective_velocity_norm",
            "projective_curvature_norm",
            "transition_score",
            "transition_score_z",
            "lie_transition_score",
        ]
        for column in expected_columns:
            self.assertIn(column, frame.columns)
        finite_expected = [
            "hmm_transition_score_z",
            "projective_level_norm_z",
            "projective_velocity_norm_z",
            "projective_curvature_norm_z",
            "transition_score_z",
            "lie_transition_score",
        ]
        for column in finite_expected:
            self.assertTrue(np.isfinite(frame[column].to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
