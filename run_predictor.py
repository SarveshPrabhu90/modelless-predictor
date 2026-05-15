"""
Entry point: collect observations, train, predict, and optimize.

Requires a running plant server on 127.0.0.1:9100
(e.g. from the explicit-model project).
"""

import os
import time

import numpy as np
from src.data_collector import collect
from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize
from src.manifest import write_manifest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    np.random.seed(42)
    t_start = time.time()

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

    # ── Write run manifest ──────────────────────────────────────────────
    duration = time.time() - t_start
    manifest_path = write_manifest(
        OUTPUT_DIR,
        analysis_type="predictor",
        data_type="synthetic",
        explicit_model_source="external TCP server (127.0.0.1:9100)",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": train_in.shape[0], "test": test_in.shape[0]},
        train_test_split={"train": 200, "test": 50},
        random_seed=42,
        noise_level=0.5,
        constraints_used={"min_purity": 90.0},
        metrics={
            "yield_r2": float(metrics["yield"]["r2"]),
            "purity_r2": float(metrics["purity"]["r2"]),
            "yield_mae": float(metrics["yield"]["mae"]),
            "purity_mae": float(metrics["purity"]["mae"]),
        },
        learned_coefficients=predictor.coefficients,
        optimization={
            "success": result.get("success", False),
            "optimal_inputs": result.get("optimal_inputs"),
            "predicted_yield": result.get("predicted_yield"),
            "predicted_purity": result.get("predicted_purity"),
        },
        all_checks_pass=None,
        plot_files=[],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
