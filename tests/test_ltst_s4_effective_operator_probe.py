from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact.ltst_s4_effective_operator_probe import (
    anti_hermitian_split,
    bilinear_discretize_dense,
    compute_global_operator_diagnostics,
    compute_effective_operator_metrics,
    derive_imminent_onset_labels,
    legs_dense_operator,
    legs_skew_matrix,
    rank_one_vector,
)


class LtstS4EffectiveOperatorProbeTests(unittest.TestCase):
    def test_legs_dense_operator_matches_shift_skew_rank1_decomposition(self) -> None:
        state_size = 6
        p = rank_one_vector(state_size)
        skew = legs_skew_matrix(state_size)
        dense = legs_dense_operator(state_size, real_shift=-0.5, skew_scale=1.0, low_rank_scale=1.0)
        expected = (-0.5 * np.eye(state_size)) + skew - (0.5 * np.outer(p, p))
        np.testing.assert_allclose(dense.real, expected, atol=1e-10)
        np.testing.assert_allclose(dense.imag, 0.0, atol=1e-10)

    def test_imminent_onset_labels_exclude_inside_event_rows(self) -> None:
        st_event = np.asarray([False, False, False, True, True, False, False, True], dtype=bool)
        st_entry = np.asarray([False, False, True, False, False, False, True, False], dtype=bool)
        labels = derive_imminent_onset_labels(st_event=st_event, st_entry=st_entry, horizon_beats=2)
        expected = np.asarray([True, True, False, False, False, True, False, False], dtype=bool)
        np.testing.assert_array_equal(labels, expected)

    def test_projected_operator_metrics_are_finite_and_pythagorean(self) -> None:
        state_size = 8
        continuous_a = legs_dense_operator(state_size)
        continuous_b = rank_one_vector(state_size).astype(np.complex128)
        discrete_a, _ = bilinear_discretize_dense(continuous_a, continuous_b, delta=0.1)
        discrete_a_base, _ = bilinear_discretize_dense(
            legs_dense_operator(state_size, low_rank_scale=0.0),
            continuous_b,
            delta=0.1,
        )
        state = np.linspace(1.0, 2.0, state_size).astype(np.complex128) + 1j * np.linspace(0.5, 1.5, state_size)
        metrics = compute_effective_operator_metrics(discrete_a=discrete_a, discrete_a_base=discrete_a_base, state=state)
        self.assertEqual(metrics["operator_valid"], 1.0)
        self.assertGreaterEqual(metrics["operator_rho"], 0.0)
        self.assertLessEqual(metrics["operator_rho"], 1.0)
        self.assertLess(metrics["operator_pythagorean_residual"], 1e-6)

    def test_anti_hermitian_split_recomposes_matrix(self) -> None:
        matrix = np.asarray([[1 + 0j, 2 + 1j], [3 - 1j, 4 + 0j]], dtype=np.complex128)
        anti, herm = anti_hermitian_split(matrix)
        np.testing.assert_allclose(anti + herm, matrix, atol=1e-10)
        np.testing.assert_allclose(anti.conj().T, -anti, atol=1e-10)
        np.testing.assert_allclose(herm.conj().T, herm, atol=1e-10)

    def test_global_operator_diagnostics_are_bounded_and_pythagorean(self) -> None:
        state_size = 6
        operator = legs_dense_operator(state_size, real_shift=-0.5, skew_scale=1.0, low_rank_scale=0.25)
        low_rank = 0.25 * 0.5 * np.outer(rank_one_vector(state_size), rank_one_vector(state_size))
        diagnostics = compute_global_operator_diagnostics(operator_matrix=operator, low_rank_matrix=low_rank)
        self.assertGreaterEqual(diagnostics["rho"], 0.0)
        self.assertLessEqual(diagnostics["rho"], 1.0)
        self.assertGreater(diagnostics["anti_norm"], 0.0)
        self.assertGreater(diagnostics["hermitian_norm"], 0.0)
        self.assertGreater(diagnostics["low_rank_norm"], 0.0)
        self.assertLess(diagnostics["pythagorean_residual"], 1e-6)


if __name__ == "__main__":
    unittest.main()
