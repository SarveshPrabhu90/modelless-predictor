"""
Entry point: collect observations, train, predict, and optimize.

Requires a running plant server on 127.0.0.1:9100
(e.g. from the explicit-model project).
"""

import numpy as np
from src.data_collector import collect
from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize


def main():
    np.random.seed(42)

    print("Collecting 200 training observations …")
    train_in, train_out = collect(200)
    print(f"  Received {train_in.shape[0]} samples  "
          f"(inputs: {train_in.shape}, outputs: {train_out.shape})")

    print("\nCollecting 50 test observations …")
    test_in, test_out = collect(50)

    print("\nTraining modelless predictor …")
    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=["yield", "purity"])

    print("  Learned coefficients:")
    for name, c in predictor.coefficients.items():
        w = [round(v, 4) for v in c["weights"]]
        print(f"    {name:8s}  weights: {w}  intercept: {round(c['intercept'], 4)}")

    print("\nEvaluating on test data …")
    metrics = predictor.evaluate(test_in, test_out)
    for name, m in metrics.items():
        print(f"    {name:8s}  MAE = {m['mae']:.4f}   R² = {m['r2']:.6f}")

    print("\nOptimizing (max yield, purity ≥ 90%) …")
    result = optimize(predictor, min_purity=90.0)
    if result["success"]:
        xi = result["optimal_inputs"]
        print(f"  Optimal inputs  → T={xi[0]:.2f}°C  F={xi[1]:.2f}L/min  C={xi[2]:.2f}%")
        print(f"  Predicted yield → {result['predicted_yield']:.2f}%")
        print(f"  Predicted purity→ {result['predicted_purity']:.2f}%")
    else:
        print(f"  Optimization failed: {result['message']}")


if __name__ == "__main__":
    main()
