"""
Modelless Predictor — Learns to predict outputs from observed data.

Instead of relying on hand-crafted coefficients, this module takes
observed input-output pairs and fits a linear regression model.
This is the heart of the POV: demonstrating that we can make equivalent
predictions without any explicit model knowledge.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


class ModellessPredictor:
    """Data-driven predictor that learns from observed system behaviour."""

    def __init__(self):
        self._models: list[LinearRegression] = []
        self._output_names: list[str] = []
        self._is_fitted = False

    # ── training ────────────────────────────────────────────────────────

    def fit(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        output_names: list[str] | None = None,
    ):
        """
        Learn from observed input-output pairs.

        Args:
            inputs:       (N, d) array of input observations
            outputs:      (N, m) array of output observations
            output_names: optional labels for each output column
        """
        inputs = np.atleast_2d(inputs)
        outputs = np.atleast_2d(outputs)
        n_outputs = outputs.shape[1]

        self._output_names = output_names or [
            f"output_{i}" for i in range(n_outputs)
        ]
        self._models = []
        for i in range(n_outputs):
            model = LinearRegression()
            model.fit(inputs, outputs[:, i])
            self._models.append(model)

        self._is_fitted = True

    # ── inference ───────────────────────────────────────────────────────

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict outputs for new inputs using the learned model."""
        if not self._is_fitted:
            raise RuntimeError("Predictor has not been fitted yet.")
        inputs = np.atleast_2d(inputs)
        return np.column_stack([m.predict(inputs) for m in self._models])

    # ── inspection ──────────────────────────────────────────────────────

    @property
    def coefficients(self) -> dict[str, dict]:
        """Return learned weights and intercepts for every output."""
        if not self._is_fitted:
            return {}
        return {
            name: {
                "weights": model.coef_.tolist(),
                "intercept": float(model.intercept_),
            }
            for name, model in zip(self._output_names, self._models)
        }

    def evaluate(
        self, inputs: np.ndarray, true_outputs: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Compute MAE and R² against ground-truth outputs."""
        preds = self.predict(inputs)
        return {
            name: {
                "mae": float(mean_absolute_error(true_outputs[:, i], preds[:, i])),
                "r2": float(r2_score(true_outputs[:, i], preds[:, i])),
            }
            for i, name in enumerate(self._output_names)
        }
