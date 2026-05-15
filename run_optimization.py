"""
Optimization agreement evaluation.

Compares optimal input recommendations from the explicit baseline versus
the modelless predictor.  Both optimizers maximize yield subject to
purity >= threshold, using linear programming over the same operating bounds.

Outputs:
  - output/optimization_agreement.csv
  - output/run_manifest.json

Requires a running plant server on 127.0.0.1:9100.
"""

import csv
import os
import time

import numpy as np
from scipy.optimize import linprog

from src.data_collector import collect
from src.manifest import write_manifest
from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize, DEFAULT_BOUNDS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_TRAIN = 300
INPUT_NAMES = ["temperature", "flow_rate", "concentration"]
OUTPUT_NAMES = ["yield", "purity"]
MIN_PURITY = 88.0

# Ground-truth coefficients (explicit baseline) — used ONLY for the
# explicit-side optimizer, never shared with the modelless predictor.
YIELD_W = np.array([0.45, 0.30, 0.80])
YIELD_I = 12.0
PURITY_W = np.array([-0.15, 0.55, 0.35])
PURITY_I = 85.0


def _explicit_optimize():
    """LP optimization using the known explicit model coefficients."""
    c = -YIELD_W
    A_ub = [-PURITY_W]
    b_ub = [-(MIN_PURITY - PURITY_I)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=DEFAULT_BOUNDS, method="highs")
    if not result.success:
        return None
    x = result.x
    pred_yield = float(x @ YIELD_W + YIELD_I)
    pred_purity = float(x @ PURITY_W + PURITY_I)
    return {
        "inputs": x.tolist(),
        "predicted_yield": pred_yield,
        "predicted_purity": pred_purity,
    }


def main():
    np.random.seed(42)
    t_start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Train modelless predictor
    print(f"Collecting {N_TRAIN} training observations …")
    train_in, train_out = collect(N_TRAIN)

    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=OUTPUT_NAMES)

    # Run both optimizers
    explicit_opt = _explicit_optimize()
    learned_opt = optimize(predictor, min_purity=MIN_PURITY)

    if explicit_opt is None or not learned_opt["success"]:
        print("ERROR: One or both optimizers failed.")
        return

    exp = explicit_opt
    lrn = learned_opt

    # Compute differences
    diffs = {}
    for i, name in enumerate(INPUT_NAMES):
        diffs[name] = abs(exp["inputs"][i] - lrn["optimal_inputs"][i])
    diffs["predicted_yield"] = abs(exp["predicted_yield"] - lrn["predicted_yield"])
    diffs["predicted_purity"] = abs(exp["predicted_purity"] - lrn["predicted_purity"])

    # Match score: 1.0 = perfect agreement, penalised by normalised input distance
    ranges = [b[1] - b[0] for b in DEFAULT_BOUNDS]
    norm_diffs = [diffs[name] / r for name, r in zip(INPUT_NAMES, ranges)]
    avg_norm_diff = sum(norm_diffs) / len(norm_diffs)
    match_score = round(max(0.0, 1.0 - avg_norm_diff), 6)

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "optimization_agreement.csv")
    row = {
        "run_id": run_id,
        "min_purity": MIN_PURITY,
        "n_train": N_TRAIN,
    }
    for i, name in enumerate(INPUT_NAMES):
        row[f"explicit_{name}"] = round(exp["inputs"][i], 6)
        row[f"learned_{name}"] = round(lrn["optimal_inputs"][i], 6)
        row[f"diff_{name}"] = round(diffs[name], 6)
    row["explicit_yield"] = round(exp["predicted_yield"], 6)
    row["learned_yield"] = round(lrn["predicted_yield"], 6)
    row["diff_yield"] = round(diffs["predicted_yield"], 6)
    row["explicit_purity"] = round(exp["predicted_purity"], 6)
    row["learned_purity"] = round(lrn["predicted_purity"], 6)
    row["diff_purity"] = round(diffs["predicted_purity"], 6)
    row["optimization_match_score"] = match_score

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"  CSV saved to: {csv_path}")

    # Print summary
    print()
    print(f"  {'':>20s}  {'Explicit':>12s}  {'Learned':>12s}  {'Diff':>10s}")
    print("  " + "-" * 58)
    for i, name in enumerate(INPUT_NAMES):
        print(f"  {name:>20s}  {exp['inputs'][i]:12.4f}  {lrn['optimal_inputs'][i]:12.4f}  {diffs[name]:10.4f}")
    print(f"  {'predicted_yield':>20s}  {exp['predicted_yield']:12.4f}  {lrn['predicted_yield']:12.4f}  {diffs['predicted_yield']:10.4f}")
    print(f"  {'predicted_purity':>20s}  {exp['predicted_purity']:12.4f}  {lrn['predicted_purity']:12.4f}  {diffs['predicted_purity']:10.4f}")
    print(f"\n  optimization_match_score = {match_score}")

    # Write manifest
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="optimization_agreement",
        data_type="synthetic",
        explicit_model_source="known coefficients",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": N_TRAIN},
        train_test_split=None,
        random_seed=42,
        noise_level=0.5,
        constraints_used={"min_purity": MIN_PURITY},
        metrics={"optimization_match_score": match_score, "diffs": diffs},
        all_checks_pass=match_score >= 0.95,
        plot_files=[],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=["output/optimization_agreement.csv"],
        duration_seconds=round(duration, 2),
    )
    print(f"  Manifest: {os.path.join(OUTPUT_DIR, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
