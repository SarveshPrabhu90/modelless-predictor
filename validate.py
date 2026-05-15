"""
Standalone validation: verify the modelless predictor's statistical soundness
against the live explicit-model TCP server.

Produces:
  - Console summary of metrics
  - 6-panel validation dashboard (validation_dashboard.png)
"""

import json
import os
import socket
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the standalone project to the path
sys.path.insert(0, r"c:\Users\sarve\OneDrive\Documents\Pilot-Projects\modelless-predictor-standalone")

from src.data_collector import collect
from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize
from src.manifest import write_manifest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _fetch_noise_free(inputs: np.ndarray, host="127.0.0.1", port=9100) -> np.ndarray:
    """Get noise-free ground-truth outputs by computing from known coefficients.

    Since the explicit model's coefficients are public in its repo, we
    replicate the deterministic calculation here for validation only.
    """
    # These match explicit_model.py exactly
    YIELD_W = np.array([0.45, 0.30, 0.80])
    YIELD_I = 12.0
    PURITY_W = np.array([-0.15, 0.55, 0.35])
    PURITY_I = 85.0
    y = inputs @ YIELD_W + YIELD_I
    p = inputs @ PURITY_W + PURITY_I
    return np.column_stack([y, p])


def main():
    np.random.seed(42)
    t_start = time.time()

    # ── 1. Collect data from the live server ────────────────────────────
    print("=" * 65)
    print("  STANDALONE MODELLESS PREDICTOR — VALIDATION RUN")
    print("=" * 65)

    print("\n  Collecting 300 training observations from TCP server …")
    train_in, train_out = collect(300)

    print("  Collecting 100 test observations …")
    test_in, test_out = collect(100)

    print(f"  Train: {train_in.shape[0]} samples | Test: {test_in.shape[0]} samples\n")

    # ── 2. Train the modelless predictor ────────────────────────────────
    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=["yield", "purity"])

    coeffs = predictor.coefficients
    print("  Learned coefficients:")
    print(f"    {'':20s}  {'weight_T':>10s} {'weight_F':>10s} {'weight_C':>10s} {'intercept':>10s}")
    for name, c in coeffs.items():
        w = c["weights"]
        print(f"    {name:20s}  {w[0]:10.4f} {w[1]:10.4f} {w[2]:10.4f} {c['intercept']:10.4f}")
    print()

    # Known ground-truth coefficients
    print("  Ground-truth coefficients (from explicit-model repo):")
    print(f"    {'':20s}  {'weight_T':>10s} {'weight_F':>10s} {'weight_C':>10s} {'intercept':>10s}")
    print(f"    {'yield':20s}  {0.45:10.4f} {0.30:10.4f} {0.80:10.4f} {12.0:10.4f}")
    print(f"    {'purity':20s}  {-0.15:10.4f} {0.55:10.4f} {0.35:10.4f} {85.0:10.4f}")
    print()

    # Coefficient recovery error
    true_yield_w = [0.45, 0.30, 0.80]
    true_purity_w = [-0.15, 0.55, 0.35]
    yield_w_err = np.abs(np.array(coeffs["yield"]["weights"]) - true_yield_w)
    purity_w_err = np.abs(np.array(coeffs["purity"]["weights"]) - true_purity_w)
    print("  Coefficient recovery error (|learned - true|):")
    print(f"    yield  weights: {yield_w_err.tolist()}  intercept: {abs(coeffs['yield']['intercept'] - 12.0):.6f}")
    print(f"    purity weights: {purity_w_err.tolist()}  intercept: {abs(coeffs['purity']['intercept'] - 85.0):.6f}")
    print()

    # ── 3. Evaluate on test data ────────────────────────────────────────
    # Compare against noisy server outputs (what we'd see in production)
    noisy_metrics = predictor.evaluate(test_in, test_out)

    # Compare against noise-free ground truth
    gt_outputs = _fetch_noise_free(test_in)
    gt_metrics = predictor.evaluate(test_in, gt_outputs)

    learned_preds = predictor.predict(test_in)

    print("  Test metrics vs NOISY observations (practical accuracy):")
    for name, m in noisy_metrics.items():
        print(f"    {name:8s}  MAE = {m['mae']:.4f}   R² = {m['r2']:.6f}")

    print("\n  Test metrics vs NOISE-FREE ground truth (theoretical accuracy):")
    for name, m in gt_metrics.items():
        print(f"    {name:8s}  MAE = {m['mae']:.4f}   R² = {m['r2']:.6f}")
    print()

    # ── 4. Residual analysis ────────────────────────────────────────────
    residuals_gt = learned_preds - gt_outputs
    residuals_noisy = learned_preds - test_out

    print("  Residual analysis (learned − noise-free ground truth):")
    for i, name in enumerate(["yield", "purity"]):
        r = residuals_gt[:, i]
        print(f"    {name:8s}  mean={r.mean():+.4f}  std={r.std():.4f}  "
              f"min={r.min():.4f}  max={r.max():.4f}")
    print()

    # ── 5. Optimization ────────────────────────────────────────────────
    print("  LP Optimization (max yield, purity ≥ 88%):")
    opt = optimize(predictor, min_purity=88.0)
    if opt["success"]:
        xi = opt["optimal_inputs"]
        print(f"    Optimal T={xi[0]:.2f}°C  F={xi[1]:.2f}L/min  C={xi[2]:.2f}%")
        print(f"    Predicted yield={opt['predicted_yield']:.2f}%  purity={opt['predicted_purity']:.2f}%")

        # Verify against ground truth
        gt_at_opt = _fetch_noise_free(np.array([xi]))
        print(f"    Ground-truth yield={gt_at_opt[0,0]:.2f}%  purity={gt_at_opt[0,1]:.2f}%")
        print(f"    Yield error: {abs(opt['predicted_yield'] - gt_at_opt[0,0]):.4f}%")
    print()

    # ── 6. Generate dashboard ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Modelless Predictor — Statistical Validation Dashboard",
                 fontsize=14, fontweight="bold")

    # Panel 1: Yield — predicted vs ground truth
    ax = axes[0, 0]
    ax.scatter(gt_outputs[:, 0], learned_preds[:, 0], s=20, alpha=0.6, c="#1f77b4")
    lo, hi = gt_outputs[:, 0].min(), gt_outputs[:, 0].max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Ground Truth Yield (%)")
    ax.set_ylabel("Predicted Yield (%)")
    ax.set_title(f"Yield: Predicted vs Truth\nR²={gt_metrics['yield']['r2']:.6f}  MAE={gt_metrics['yield']['mae']:.4f}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    # Panel 2: Purity — predicted vs ground truth
    ax = axes[0, 1]
    ax.scatter(gt_outputs[:, 1], learned_preds[:, 1], s=20, alpha=0.6, c="#ff7f0e")
    lo, hi = gt_outputs[:, 1].min(), gt_outputs[:, 1].max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Ground Truth Purity (%)")
    ax.set_ylabel("Predicted Purity (%)")
    ax.set_title(f"Purity: Predicted vs Truth\nR²={gt_metrics['purity']['r2']:.6f}  MAE={gt_metrics['purity']['mae']:.4f}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    # Panel 3: Coefficient comparison bar chart
    ax = axes[0, 2]
    labels = ["T (yield)", "F (yield)", "C (yield)", "T (purity)", "F (purity)", "C (purity)"]
    true_vals = [0.45, 0.30, 0.80, -0.15, 0.55, 0.35]
    learned_vals = coeffs["yield"]["weights"] + coeffs["purity"]["weights"]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, true_vals, width, label="True", color="#2ca02c", alpha=0.8)
    ax.bar(x_pos + width / 2, learned_vals, width, label="Learned", color="#d62728", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient value")
    ax.set_title("Coefficient Recovery")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)

    # Panel 4: Yield residual histogram
    ax = axes[1, 0]
    ax.hist(residuals_gt[:, 0], bins=25, color="#1f77b4", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (predicted − truth)")
    ax.set_ylabel("Count")
    r_y = residuals_gt[:, 0]
    ax.set_title(f"Yield Residuals\nμ={r_y.mean():+.4f}  σ={r_y.std():.4f}")

    # Panel 5: Purity residual histogram
    ax = axes[1, 1]
    ax.hist(residuals_gt[:, 1], bins=25, color="#ff7f0e", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (predicted − truth)")
    ax.set_ylabel("Count")
    r_p = residuals_gt[:, 1]
    ax.set_title(f"Purity Residuals\nμ={r_p.mean():+.4f}  σ={r_p.std():.4f}")

    # Panel 6: Residual Q-Q style — sorted residuals vs expected normal
    ax = axes[1, 2]
    for i, (name, color) in enumerate(zip(["Yield", "Purity"], ["#1f77b4", "#ff7f0e"])):
        r = np.sort(residuals_gt[:, i])
        n = len(r)
        theoretical = np.linspace(-2.5, 2.5, n)
        ax.scatter(theoretical, r, s=12, alpha=0.6, c=color, label=name)
    ax.plot([-3, 3], [-3, 3], "r--", lw=1, alpha=0.5, label="Normal ref")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Residual quantiles")
    ax.set_title("Residual Normality Check")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "validation_dashboard.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Dashboard saved to: {out_path}")
    plt.show()

    # ── 7. Pass/fail summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  VALIDATION SUMMARY")
    print("=" * 65)
    checks = []
    checks.append(("Yield R² ≥ 0.999 (vs ground truth)", gt_metrics["yield"]["r2"] >= 0.999))
    checks.append(("Purity R² ≥ 0.999 (vs ground truth)", gt_metrics["purity"]["r2"] >= 0.999))
    checks.append(("Yield MAE < 0.1", gt_metrics["yield"]["mae"] < 0.1))
    checks.append(("Purity MAE < 0.1", gt_metrics["purity"]["mae"] < 0.1))
    checks.append(("Yield residual mean ≈ 0 (|μ| < 0.1)", abs(residuals_gt[:, 0].mean()) < 0.1))
    checks.append(("Purity residual mean ≈ 0 (|μ| < 0.1)", abs(residuals_gt[:, 1].mean()) < 0.1))
    checks.append(("Optimizer finds feasible solution", opt.get("success", False)))
    checks.append(("Optimizer yield error < 0.5%",
                    opt["success"] and abs(opt["predicted_yield"] - gt_at_opt[0, 0]) < 0.5))

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        marker = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"  {marker} {status}  {desc}")

    print()
    if all_pass:
        print("  ✓ ALL CHECKS PASSED — Standalone modelless predictor is statistically sound.")
    else:
        print("  ✗ SOME CHECKS FAILED — Review the dashboard for details.")
    print()

    # ── 8. Write run manifest ──────────────────────────────────────────
    duration = time.time() - t_start
    manifest_path = write_manifest(
        OUTPUT_DIR,
        analysis_type="predictor",
        data_type="synthetic",
        explicit_model_source="external TCP server (127.0.0.1:9100)",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": train_in.shape[0], "test": test_in.shape[0]},
        train_test_split={"train": 300, "test": 100},
        random_seed=42,
        noise_level=0.5,
        constraints_used={"min_purity": 88.0},
        metrics={
            "yield_r2_vs_truth": float(gt_metrics["yield"]["r2"]),
            "purity_r2_vs_truth": float(gt_metrics["purity"]["r2"]),
            "yield_mae_vs_truth": float(gt_metrics["yield"]["mae"]),
            "purity_mae_vs_truth": float(gt_metrics["purity"]["mae"]),
            "yield_r2_vs_noisy": float(noisy_metrics["yield"]["r2"]),
            "purity_r2_vs_noisy": float(noisy_metrics["purity"]["r2"]),
        },
        learned_coefficients=coeffs,
        optimization={
            "success": opt.get("success", False),
            "optimal_inputs": opt.get("optimal_inputs"),
            "predicted_yield": opt.get("predicted_yield"),
            "predicted_purity": opt.get("predicted_purity"),
        },
        all_checks_pass=all_pass,
        plot_files=["output/validation_dashboard.png"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )
    print(f"  Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
