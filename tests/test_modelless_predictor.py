"""Unit tests for the ModellessPredictor."""

import numpy as np
import pytest

from src.modelless_predictor import ModellessPredictor


@pytest.fixture()
def trained_predictor():
    """Return a predictor trained on a known linear system."""
    np.random.seed(0)
    n = 200
    inputs = np.random.rand(n, 3) * [60, 9, 5] + [20, 1, 0.1]
    # Known linear relationship: y = 0.5*x0 + 0.3*x1 + 0.8*x2 + 10
    #                            p = -0.2*x0 + 0.6*x1 + 0.4*x2 + 80
    outputs = np.column_stack([
        inputs @ [0.5, 0.3, 0.8] + 10,
        inputs @ [-0.2, 0.6, 0.4] + 80,
    ])
    pred = ModellessPredictor()
    pred.fit(inputs, outputs, output_names=["yield", "purity"])
    return pred, inputs, outputs


class TestFit:
    def test_is_fitted(self, trained_predictor):
        pred, _, _ = trained_predictor
        assert pred._is_fitted is True

    def test_output_names(self, trained_predictor):
        pred, _, _ = trained_predictor
        assert pred._output_names == ["yield", "purity"]

    def test_default_output_names(self):
        pred = ModellessPredictor()
        inputs = np.random.rand(20, 2)
        outputs = np.random.rand(20, 2)
        pred.fit(inputs, outputs)
        assert pred._output_names == ["output_0", "output_1"]

    def test_model_count(self, trained_predictor):
        pred, _, _ = trained_predictor
        assert len(pred._models) == 2


class TestPredict:
    def test_shape(self, trained_predictor):
        pred, _, _ = trained_predictor
        out = pred.predict(np.array([[50.0, 5.0, 2.5]]))
        assert out.shape == (1, 2)

    def test_accuracy_on_training_data(self, trained_predictor):
        """Predictions should match the noise-free training data closely."""
        pred, inputs, outputs = trained_predictor
        preds = pred.predict(inputs)
        np.testing.assert_allclose(preds, outputs, atol=1e-10)

    def test_known_prediction(self, trained_predictor):
        pred, _, _ = trained_predictor
        inp = np.array([[50.0, 5.0, 2.5]])
        out = pred.predict(inp)
        expected_yield = 50.0 * 0.5 + 5.0 * 0.3 + 2.5 * 0.8 + 10
        expected_purity = 50.0 * (-0.2) + 5.0 * 0.6 + 2.5 * 0.4 + 80
        assert out[0, 0] == pytest.approx(expected_yield, abs=1e-6)
        assert out[0, 1] == pytest.approx(expected_purity, abs=1e-6)

    def test_raises_if_not_fitted(self):
        pred = ModellessPredictor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            pred.predict(np.array([[1, 2, 3]]))

    def test_1d_input_promoted(self, trained_predictor):
        pred, _, _ = trained_predictor
        out = pred.predict(np.array([50.0, 5.0, 2.5]))
        assert out.shape == (1, 2)


class TestCoefficients:
    def test_returns_dict_when_fitted(self, trained_predictor):
        pred, _, _ = trained_predictor
        coeffs = pred.coefficients
        assert "yield" in coeffs
        assert "purity" in coeffs

    def test_learned_weights_close(self, trained_predictor):
        pred, _, _ = trained_predictor
        coeffs = pred.coefficients
        np.testing.assert_allclose(coeffs["yield"]["weights"], [0.5, 0.3, 0.8], atol=1e-10)
        np.testing.assert_allclose(coeffs["purity"]["weights"], [-0.2, 0.6, 0.4], atol=1e-10)

    def test_learned_intercepts_close(self, trained_predictor):
        pred, _, _ = trained_predictor
        coeffs = pred.coefficients
        assert coeffs["yield"]["intercept"] == pytest.approx(10.0, abs=1e-8)
        assert coeffs["purity"]["intercept"] == pytest.approx(80.0, abs=1e-8)

    def test_empty_when_not_fitted(self):
        pred = ModellessPredictor()
        assert pred.coefficients == {}


class TestEvaluate:
    def test_perfect_metrics(self, trained_predictor):
        pred, inputs, outputs = trained_predictor
        metrics = pred.evaluate(inputs, outputs)
        assert metrics["yield"]["r2"] == pytest.approx(1.0, abs=1e-8)
        assert metrics["purity"]["r2"] == pytest.approx(1.0, abs=1e-8)
        assert metrics["yield"]["mae"] == pytest.approx(0.0, abs=1e-8)
        assert metrics["purity"]["mae"] == pytest.approx(0.0, abs=1e-8)

    def test_metric_keys(self, trained_predictor):
        pred, inputs, outputs = trained_predictor
        metrics = pred.evaluate(inputs, outputs)
        for name in ["yield", "purity"]:
            assert "mae" in metrics[name]
            assert "r2" in metrics[name]
