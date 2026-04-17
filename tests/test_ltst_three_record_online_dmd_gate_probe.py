import unittest

import numpy as np
import pandas as pd

from artifact.ltst_three_record_online_dmd_gate_probe import (
    GateProbeConfig,
    add_gate_scores,
    add_multiscale_gate_scores,
    add_routed_models,
    build_probe_frame,
    evaluate_models,
)


class LTSTThreeRecordOnlineDMDGateProbeTest(unittest.TestCase):
    def test_add_gate_scores_builds_expected_directionality(self) -> None:
        panel = pd.DataFrame(
            {
                "patient_id": ["sA"] * 6,
                "time_seconds": [-5, -4, -3, -2, -1, 0],
                "dmd_observer_lie_metric_drift": [0, 1, 2, 3, 4, 5],
                "dmd_observer_pair_angle_sl": [0, 1, 2, 3, 4, 5],
                "dmd_observer_memory_align": [5, 4, 3, 2, 1, 0],
                "dmd_observer_lie_transition_score": [0, 0, 0, 0.5, 0.5, 0.5],
                "dmd_observer_lie_orbit_norm": [0, 0, 0, 0.5, 0.5, 0.5],
                "dmd_observer_lie_strain_norm": [0, 0, 0, 0.5, 0.5, 0.5],
                "dmd_observer_combined_transition_score": [0, 0, 0, 0.5, 0.5, 0.5],
            }
        )
        scored = add_gate_scores(panel, bg_rows=3)
        self.assertIn("observer_gate_score", scored.columns)
        self.assertIn("observer_selective_margin", scored.columns)
        self.assertGreater(scored.loc[5, "observer_gate_score"], scored.loc[0, "observer_gate_score"])
        self.assertGreaterEqual(scored.loc[5, "observer_gate_relu"], 0.0)
        multiscale = add_multiscale_gate_scores(scored, fast_rows=2, short_rows=4, long_rows=6)
        self.assertIn("fast_gate_score", multiscale.columns)
        self.assertIn("short_selective_margin", multiscale.columns)
        self.assertIn("long_gate_score", multiscale.columns)

    def test_route_max_uses_observer_only_when_gate_active(self) -> None:
        df = pd.DataFrame(
            {
                "substrate_only": [0.2, 0.2, 0.2],
                "observer_only": [0.1, 0.8, 0.9],
                "hybrid_only": [0.15, 0.5, 0.5],
                "observer_gate_score": [-0.1, 0.6, 0.6],
                "observer_gate_relu": [0.0, 0.6, 0.6],
                "fast_gate_score": [-0.1, 0.6, 0.6],
                "short_gate_score": [-0.1, 0.6, 0.6],
                "long_gate_score": [-0.1, 0.6, 0.6],
                "fast_selective_margin": [-0.1, 0.6, 0.6],
                "short_selective_margin": [-0.1, 0.6, 0.6],
                "long_selective_margin": [-0.1, 0.6, 0.6],
                "observer_selective_margin": [-0.1, 0.6, 0.6],
            }
        )
        routed = add_routed_models(df, thresholds=(0.5,))
        self.assertAlmostEqual(routed.loc[0, "route_max_t0p5"], 0.2)
        self.assertAlmostEqual(routed.loc[1, "route_max_t0p5"], 0.8)
        self.assertAlmostEqual(routed.loc[2, "route_switch_t0p5"], 0.9)
        self.assertAlmostEqual(routed.loc[1, "route_multiscale_switch_t0p5"], 0.8)
        self.assertAlmostEqual(routed.loc[1, "route_vhr_switch_t0p5"], 0.8)

    def test_build_probe_frame_and_metrics(self) -> None:
        panel = pd.DataFrame(
            {
                "patient_id": ["sA", "sA", "sB", "sB"],
                "time_seconds": [0.0, 1.0, 0.0, 1.0],
                "dmd_observer_lie_metric_drift": [0.0, 2.0, 0.0, 0.5],
                "dmd_observer_pair_angle_sl": [0.0, 2.0, 0.0, 0.2],
                "dmd_observer_memory_align": [2.0, 0.0, 2.0, 1.8],
                "dmd_observer_lie_transition_score": [0.0, 0.3, 0.0, 1.5],
                "dmd_observer_lie_orbit_norm": [0.0, 0.3, 0.0, 1.5],
                "dmd_observer_lie_strain_norm": [0.0, 0.3, 0.0, 1.5],
                "dmd_observer_combined_transition_score": [0.0, 0.3, 0.0, 1.5],
                "future_onset_within_5": [0, 1, 0, 1],
            }
        )
        preds = pd.DataFrame(
            {
                "patient_id": ["sA", "sA", "sA", "sA", "sB", "sB", "sB", "sB"],
                "time_seconds": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "horizon": [5, 5, 5, 5, 5, 5, 5, 5],
                "model_name": [
                    "substrate_only",
                    "observer_only",
                    "substrate_only",
                    "observer_only",
                    "substrate_only",
                    "observer_only",
                    "substrate_only",
                    "observer_only",
                ],
                "prediction": [0.2, 0.1, 0.3, 0.9, 0.3, 0.2, 0.4, 0.1],
            }
        )
        probe = build_probe_frame(panel, preds, GateProbeConfig(bg_rows=1))
        probe["hybrid_only"] = np.nan
        probe = add_routed_models(probe, thresholds=(0.0,), selective_margin_threshold=0.0)
        patient_metrics, overall_metrics = evaluate_models(probe)
        self.assertFalse(patient_metrics.empty)
        self.assertFalse(overall_metrics.empty)
        self.assertIn("route_max_t0p0", patient_metrics["model_name"].tolist())
        self.assertIn("route_selective_max_t0p0", patient_metrics["model_name"].tolist())
        self.assertIn("route_multiscale_max_t0p0", patient_metrics["model_name"].tolist())
        self.assertIn("route_vhr_max_t0p0", patient_metrics["model_name"].tolist())


if __name__ == "__main__":
    unittest.main()
