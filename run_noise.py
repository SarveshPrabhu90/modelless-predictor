"""
Noise and process variation sensitivity test.

Trains the modelless predictor at different noise levels to measure how
robust the learned model is to observation noise.  At each level, the
script generates synthetic data using the ground-truth coefficients
(client-side) plus Gaussian noise, trains on it, and evaluates against
the noise-free ground truth.

Noise levels:
  - none   (σ = 0.0)
  - low    (σ = 0.25)
  - medium (σ = 0.5)   ← matches the actual server
  - high   (σ = 1.0)
  - extreme(σ = 2.0)

Outputs:
  - output/noise_sensitivity.csv
  - output/run_manifest.json

Does NOT require a running TCP server (generates data locally).
"""

import csv
import os
import time

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.manifest import write_manifest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_TRAIN = 300
N_TEST = 200
INPUT_NAMES = ["temperature", "flow_rate", "concentration"]
OUTPUT_NAMES = ["yield", "purity"]
INPUT_RANGES = [(20.0, 80.0), (1.0, 10.0), (0.1, 5.0)]
MIN_PURITY = 88.0

NOISE_LEVELS = [
    ("none",    0.0),
    ("low",     0.25),
    ("medium",  0.5),
    ("high",    1.0),
    ("extreme", 2.0),
]

# Ground-truth coefficients (used for data generation and evaluation)
YIELD_W = np.array([0.45, 0.30, 0.80])
YIELD_I = 12.0
PURITY_W = np.array([-0.15, 0.55, 0.35])
PURITY_I = 85.0


def _generate(n, noise_std, rng):
    """Generate n observations with specified noise level."""
    inputs = np.column_stack([
        rng.uniform(lo, hi, n) for lo, hi in INPUT_RANGES
    ])
    gt_yield = inputs @ YIELD_W + YIELD_I
    gt_purity = inputs @ PURITY_W + PURITY_I
    noisy_yield = gt_yield + rng.normal(0, noise_std, n) if noise_std > 0 else gt_yield.copy()
    noisy_purity = gt_purity + rng.normal(0, noise_std, n) if noise_std > 0 else gt_purity.copy()
    outputs = np.column_stack([noisy_yield, noisy_purity])
    gt = np.column_stack([gt_yield, gt_purity])
    return inputs, outputs, gt


def _optimize_learned(models, min_purity):
    """LP optimization using learned coefficients."""
    yield_w = np.array(models[0].coef_)
    purity_w = np.array(models[1].coef_)
    purity_i = float(models[1].intercept_)
    c = -yield_w
    A_ub = [-purity_w]
    b_ub = [-(min_purity - purity_i)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=INPUT_RANGES, method="highs")
    if result.success:
        x = result.x
        pred = np.column_stack([m.predict(x.reshape(1, -1)) for m in models])[0]
        return {"success": True, "inputs": x, "yield": float(pred[0]), "purity": float(pred[1])}
    return {"success": False}


def _optimize_explicit(min_purity):
    """LP optimization using known coefficients."""
    c = -YIELD_W
    A_ub = [-PURITY_W]
    b_ub = [-(min_purity - PURITY_I)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=INPUT_RANGES, method="highs")
    if result.success:
        x = result.x
        return {"success": True, "inputs": x,
                "yield": float(x @ YIELD_W + YIELD_I),
                "purity": float(x @ PURITY_W + PURITY_I)}
    return {"success": False}


def main():
    rng = np.random.default_rng(42)
    t_start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Fixed test set (noise-free ground truth)
    test_in, _, gt_test = _generate(N_TEST, 0.0, rng)

    # Explicit optimum (reference)
    exp_opt = _optimize_explicit(MIN_PURITY)

    rows = []
    header = (f"  {'level':>8s}  {'sigma':>5s}"
              f"  {'y_mae':>8s} {'y_rmse':>8s} {'y_r2':>10s}"
              f"  {'p_mae':>8s} {'p_rmse':>8s} {'p_r2':>10s}"
              f"  {'match':>6s}  {'safe':>5s}")
    print(header)
    print("  " + "-" * 90)

    for label, sigma in NOISE_LEVELS:
        train_in, train_out, _ = _generate(N_TRAIN, sigma, rng)

        # Train
        models = []
        for i in range(2):
            m = LinearRegression()
            m.fit(train_in, train_out[:, i])
            models.append(m)

        # Evaluate on noise-free test set
        preds = np.column_stack([m.predict(test_in) for m in models])
        row = {"run_id": run_id, "noise_label": label, "noise_std": sigma}
        for i, name in enumerate(OUTPUT_NAMES):
            row[f"{name}_mae"] = round(float(mean_absolute_error(gt_test[:, i], preds[:, i])), 6)
            row[f"{name}_rmse"] = round(float(np.sqrt(mean_squared_error(gt_test[:, i], preds[:, i]))), 6)
            row[f"{name}_r2"] = round(float(r2_score(gt_test[:, i], preds[:, i])), 6)

        # Optimization agreement
        lrn_opt = _optimize_learned(models, MIN_PURITY)
        if lrn_opt["success"] and exp_opt["success"]:
            ranges = [b[1] - b[0] for b in INPUT_RANGES]
            norm_diffs = [abs(exp_opt["inputs"][j] - lrn_opt["inputs"][j]) / r
                          for j, r in enumerate(ranges)]
            match_score = round(max(0.0, 1.0 - sum(norm_diffs) / len(norm_diffs)), 6)
        else:
            match_score = 0.0
        row["opt_match_score"] = match_score

        # Constraint check: does the learned recommendation satisfy baseline purity?
        if lrn_opt["success"]:
            gt_at_rec = float(lrn_opt["inputs"] @ PURITY_W + PURITY_I)
            row["constraint_safe"] = gt_at_rec >= MIN_PURITY
        else:
            row["constraint_safe"] = False

        rows.append(row)

        print(f"  {label:>8s}  {sigma:5.2f}"
              f"  {row['yield_mae']:8.4f} {row['yield_rmse']:8.4f} {row['yield_r2']:10.6f}"
              f"  {row['purity_mae']:8.4f} {row['purity_rmse']:8.4f} {row['purity_r2']:10.6f}"
              f"  {match_score:6.4f}  {str(row['constraint_safe']):>5s}")

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "noise_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved to: {csv_path}")

    # Write manifest
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="noise_sensitivity",
        data_type="synthetic_variable_noise",
        explicit_model_source="known coefficients (client-side)",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": N_TRAIN, "test": N_TEST},
        train_test_split={"train": N_TRAIN, "test": N_TEST},
        random_seed=42,
        noise_level={label: sigma for label, sigma in NOISE_LEVELS},
        constraints_used={"min_purity": MIN_PURITY},
        metrics={r["noise_label"]: {k: r[k] for k in r if k not in ("run_id", "noise_label")}
                 for r in rows},
        all_checks_pass=all(r["constraint_safe"] for r in rows),
        plot_files=[],
        metric_files=["output/noise_sensitivity.csv"],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )
    print(f"  Manifest: {os.path.join(OUTPUT_DIR, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
