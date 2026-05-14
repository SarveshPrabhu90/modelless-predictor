"""
Optimizer — Finds optimal control inputs via Linear Programming.

Uses the learned (modelless) model's coefficients to drive LP optimization.

Problem:  maximise **yield** subject to **purity ≥ threshold**
          within the valid operating ranges for each input variable.
"""

import numpy as np
from scipy.optimize import linprog

from .modelless_predictor import ModellessPredictor

# Default operating bounds (must match the plant server's ranges)
DEFAULT_BOUNDS = [
    (20.0, 80.0),   # temperature (°C)
    (1.0, 10.0),    # flow_rate (L/min)
    (0.1, 5.0),     # concentration (%)
]


def optimize(
    predictor: ModellessPredictor,
    min_purity: float = 90.0,
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Maximise yield using coefficients the predictor learned from data.

    Args:
        predictor:  a fitted ModellessPredictor
        min_purity: minimum purity constraint (%)
        bounds:     operating bounds per input variable

    Returns:
        dict with optimal inputs, predicted outputs, and success flag
    """
    bounds = bounds or DEFAULT_BOUNDS

    coeffs = predictor.coefficients
    yield_w = np.array(coeffs["yield"]["weights"])
    purity_w = np.array(coeffs["purity"]["weights"])
    purity_i = coeffs["purity"]["intercept"]

    c = -yield_w  # negate for minimisation
    A_ub = [-purity_w]
    b_ub = [-(min_purity - purity_i)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        x = result.x
        predicted = predictor.predict(x.reshape(1, -1))
        return {
            "success": True,
            "optimal_inputs": x.tolist(),
            "predicted_yield": float(predicted[0, 0]),
            "predicted_purity": float(predicted[0, 1]),
        }
    return {"success": False, "message": result.message}
