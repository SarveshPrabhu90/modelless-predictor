"""Unit tests for the LP optimizer."""

import numpy as np
import pytest

from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize, DEFAULT_BOUNDS


@pytest.fixture()
def trained_predictor():
    """Predictor trained on a known linear system (no noise).

    Uses the same coefficients as the explicit model so that purity >= 90
    is feasible (max achievable purity ~ 93 with these ranges).
    """
    np.random.seed(0)
    n = 300
    inputs = np.random.rand(n, 3) * [60, 9, 5] + [20, 1, 0.1]
    # Coefficients match the explicit model:
    #   yield  = 0.45*T + 0.30*F + 0.80*C + 12
    #   purity = -0.15*T + 0.55*F + 0.35*C + 85
    # Max purity: -0.15*20 + 0.55*10 + 0.35*5 + 85 = 89.25
    # Use min_purity=88 in tests to stay feasible
    outputs = np.column_stack([
        inputs @ [0.45, 0.30, 0.80] + 12,
        inputs @ [-0.15, 0.55, 0.35] + 85,
    ])
    pred = ModellessPredictor()
    pred.fit(inputs, outputs, output_names=["yield", "purity"])
    return pred


class TestOptimize:
    def test_success(self, trained_predictor):
        result = optimize(trained_predictor, min_purity=88.0)
        assert result["success"] is True

    def test_has_required_keys(self, trained_predictor):
        result = optimize(trained_predictor, min_purity=88.0)
        assert "optimal_inputs" in result
        assert "predicted_yield" in result
        assert "predicted_purity" in result

    def test_inputs_within_bounds(self, trained_predictor):
        result = optimize(trained_predictor, min_purity=88.0)
        for val, (lo, hi) in zip(result["optimal_inputs"], DEFAULT_BOUNDS):
            assert lo - 1e-6 <= val <= hi + 1e-6

    def test_purity_meets_constraint(self, trained_predictor):
        result = optimize(trained_predictor, min_purity=88.0)
        assert result["predicted_purity"] >= 88.0 - 1e-6

    def test_yield_positive(self, trained_predictor):
        result = optimize(trained_predictor, min_purity=88.0)
        assert result["predicted_yield"] > 0

    def test_maximizes_yield(self, trained_predictor):
        """Optimal yield should be higher than a low-input feasible point."""
        result = optimize(trained_predictor, min_purity=88.0)
        # Use the minimum operating point — guaranteed to have low yield
        low_point = trained_predictor.predict(np.array([[20.0, 1.0, 0.1]]))[0, 0]
        assert result["predicted_yield"] >= low_point

    def test_tighter_purity_reduces_yield(self, trained_predictor):
        """A stricter purity constraint should give equal or lower yield."""
        r1 = optimize(trained_predictor, min_purity=85.0)
        r2 = optimize(trained_predictor, min_purity=89.0)
        assert r1["predicted_yield"] >= r2["predicted_yield"] - 1e-6

    def test_infeasible_returns_failure(self, trained_predictor):
        """An impossibly high purity constraint should fail."""
        result = optimize(trained_predictor, min_purity=999.0)
        assert result["success"] is False

    def test_custom_bounds(self, trained_predictor):
        narrow = [(40.0, 60.0), (3.0, 7.0), (1.0, 3.0)]
        result = optimize(trained_predictor, min_purity=88.0, bounds=narrow)
        if result["success"]:
            for val, (lo, hi) in zip(result["optimal_inputs"], narrow):
                assert lo - 1e-6 <= val <= hi + 1e-6
