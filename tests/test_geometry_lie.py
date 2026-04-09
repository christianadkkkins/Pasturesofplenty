from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    CANONICAL_LIE_FEATURE_COLUMNS,
    PROJECTIVE_FEATURE_COLUMNS,
    EPS,
    compute_lie_state_features,
    compute_projective_state_features,
    ema_prior_states,
)


class GeometryLieTests(unittest.TestCase):
    def test_projective_and_lie_features_have_expected_shape_and_finite_values(self) -> None:
        t = np.linspace(0.0, 1.0, 12, dtype=float)
        x = np.stack(
            [
                np.sin(2.0 * np.pi * t) + 0.1,
                np.cos(2.0 * np.pi * t) + 0.2,
                t + 0.3,
                t**2 + 0.4,
            ],
            axis=1,
        )
        x = x - x.mean(axis=1, keepdims=True)
        ms = ema_prior_states(x, beta=0.10)
        ml = ema_prior_states(x, beta=0.01)
        ms[0] = 0.5 * x[0] + 0.1
        ml[0] = 0.25 * x[0] + 0.2

        projective = compute_projective_state_features(x, ms, ml, min_state_energy=1e-12)
        lie = compute_lie_state_features(x, ms, ml, chunk_size=5)

        self.assertEqual(set(projective.keys()), set(PROJECTIVE_FEATURE_COLUMNS))
        self.assertEqual(set(lie.keys()), set(CANONICAL_LIE_FEATURE_COLUMNS))
        for values in projective.values():
            self.assertEqual(values.shape, (len(x),))
            self.assertTrue(np.isfinite(values).all())
        for values in lie.values():
            self.assertEqual(values.shape, (len(x),))
            self.assertTrue(np.isfinite(values).all())

    def test_k_skew_and_k_symmetric_decomposition_identities_hold(self) -> None:
        t = np.linspace(0.0, 1.0, 10, dtype=float)
        x = np.stack(
            [
                0.2 + np.sin(1.5 * np.pi * t),
                0.4 + np.cos(2.5 * np.pi * t),
                0.6 + t,
            ],
            axis=1,
        )
        x = x - x.mean(axis=1, keepdims=True)
        ms = ema_prior_states(x, beta=0.15)
        ml = ema_prior_states(x, beta=0.03)

        prev_x = np.zeros_like(x)
        prev_ms = np.zeros_like(ms)
        prev_ml = np.zeros_like(ml)
        prev_x[1:] = x[:-1]
        prev_ms[1:] = ms[:-1]
        prev_ml[1:] = ml[:-1]

        q = np.stack([x, ms, ml], axis=2)
        dq = np.stack([x - prev_x, ms - prev_ms, ml - prev_ml], axis=2)
        k_raw = np.matmul(np.swapaxes(q, 1, 2), q)
        lam = 1e-6 * (np.trace(k_raw, axis1=1, axis2=2) / 3.0 + EPS)
        k = k_raw + lam[:, None, None] * np.eye(3, dtype=float)[None, :, :]
        c = np.matmul(np.swapaxes(q, 1, 2), dq)
        inv_k = np.linalg.inv(k)
        b = np.matmul(inv_k, c)
        k_inv_bt_k = np.matmul(inv_k, np.matmul(np.swapaxes(b, 1, 2), k))
        a = 0.5 * (b - k_inv_bt_k)
        s = 0.5 * (b + k_inv_bt_k)

        self.assertTrue(np.allclose(b, a + s, atol=1e-10))
        for idx in range(len(x)):
            skew_residual = a[idx].T @ k[idx] + k[idx] @ a[idx]
            sym_residual = s[idx].T @ k[idx] - k[idx] @ s[idx]
            self.assertLess(float(np.max(np.abs(skew_residual))), 1e-8)
            self.assertLess(float(np.max(np.abs(sym_residual))), 1e-8)


if __name__ == "__main__":
    unittest.main()
