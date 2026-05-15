"""
Constraint verification for modelless recommendations.

Takes the modelless optimizer's recommended inputs and verifies them
against the explicit baseline (noise-free ground truth).  Checks whether
all constraints still pass under baseline verification.

Outputs:
  - output/constraint_verification.csv
  - output/run_manifest.json

Requires a running plant server on 127.0.0.1:9100.
"""

import csv
import os
import time

import numpy as np

from src.data_collector import collect
from src.manifest import write_manifest
from src.modelless_predictor import ModellessPredictor
from src.optimizer import optimize, DEFAULT_BOUNDS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_TRAIN = 300
INPUT_NAMES = ["temperature", "flow_rate", "concentration"]
OUTPUT_NAMES = ["yield", "purity"]

# Constraints to verify
CONSTRAINTS = [
    {"name": "purity_min", "output": "purity", "op": ">=", "threshold": 88.0},
    {"name": "yield_positive", "output": "yield", "op": ">", "threshold": 0.0},
    {"name": "temperature_range", "input": "temperature", "op": "in_range", "low": 20.0, "high": 80.0},
    {"name": "flow_rate_range", "input": "flow_rate", "op": "in_range", "low": 1.0, "high": 10.0},
    {"name": "concentration_range", "input": "concentration", "op": "in_range", "low": 0.1, "high": 5.0},
]

# Ground-truth coefficients — used ONLY for baseline verification
YIELD_W = np.array([0.45, 0.30, 0.80])
YIELD_I = 12.0
PURITY_W = np.array([-0.15, 0.55, 0.35])
PURITY_I = 85.0


def _ground_truth(x):
    """Noise-free output from explicit baseline."""
    y = float(x @ YIELD_W + YIELD_I)
    p = float(x @ PURITY_W + PURITY_I)
    return y, p


def _check_constraint(constraint, inputs_dict, baseline_outputs):
    """Evaluate one constraint. Returns (pass, detail_string)."""
    if "output" in constraint:
        val = baseline_outputs[constraint["output"]]
        op = constraint["op"]
        th = constraint["threshold"]
        if op == ">=":
            ok = val >= th
        elif op == ">":
            ok = val > th
        else:
            ok = False
        return ok, f"{constraint['output']}={val:.4f} {op} {th}"
    elif "input" in constraint:
        val = inputs_dict[constraint["input"]]
        ok = constraint["low"] <= val <= constraint["high"]
        return ok, f"{constraint['input']}={val:.4f} in [{constraint['low']}, {constraint['high']}]"
    return False, "unknown constraint"


def main():
    np.random.seed(42)
    t_start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Train modelless predictor
    print(f"Collecting {N_TRAIN} training observations …")
    train_in, train_out = collect(N_TRAIN)

    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=OUTPUT_NAMES)

    # Get modelless optimizer recommendation
    opt = optimize(predictor, min_purity=88.0)
    if not opt["success"]:
        print("ERROR: Modelless optimizer failed.")
        return

    rec_inputs = np.array(opt["optimal_inputs"])
    inputs_dict = {name: float(rec_inputs[i]) for i, name in enumerate(INPUT_NAMES)}
    ml_yield = opt["predicted_yield"]
    ml_purity = opt["predicted_purity"]

    # Verify against explicit baseline
    gt_yield, gt_purity = _ground_truth(rec_inputs)
    baseline_outputs = {"yield": gt_yield, "purity": gt_purity}

    # Check all constraints against baseline outputs
    results = []
    for c in CONSTRAINTS:
        passed, detail = _check_constraint(c, inputs_dict, baseline_outputs)
        results.append({
            "constraint": c["name"],
            "detail": detail,
            "pass": passed,
        })

    recommendation_safe = all(r["pass"] for r in results)

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "constraint_verification.csv")
    fieldnames = ["run_id"]
    for name in INPUT_NAMES:
        fieldnames.append(f"recommended_{name}")
    fieldnames += [
        "modelless_yield", "modelless_purity",
        "baseline_yield", "baseline_purity",
    ]
    for c in CONSTRAINTS:
        fieldnames.append(f"constraint_{c['name']}")
    fieldnames.append("recommendation_safe")

    row = {"run_id": run_id}
    for i, name in enumerate(INPUT_NAMES):
        row[f"recommended_{name}"] = round(float(rec_inputs[i]), 6)
    row["modelless_yield"] = round(ml_yield, 6)
    row["modelless_purity"] = round(ml_purity, 6)
    row["baseline_yield"] = round(gt_yield, 6)
    row["baseline_purity"] = round(gt_purity, 6)
    for r in results:
        row[f"constraint_{r['constraint']}"] = r["pass"]
    row["recommendation_safe"] = recommendation_safe

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"  CSV saved to: {csv_path}")

    # Print summary
    print()
    print(f"  Recommended inputs:")
    for name in INPUT_NAMES:
        print(f"    {name:>20s} = {inputs_dict[name]:.4f}")
    print()
    print(f"  {'':>25s}  {'Modelless':>12s}  {'Baseline':>12s}")
    print("  " + "-" * 55)
    print(f"  {'yield':>25s}  {ml_yield:12.4f}  {gt_yield:12.4f}")
    print(f"  {'purity':>25s}  {ml_purity:12.4f}  {gt_purity:12.4f}")
    print()
    print(f"  Constraint checks (against baseline):")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"    [{status}] {r['constraint']:>25s}  {r['detail']}")
    print(f"\n  recommendation_safe = {recommendation_safe}")

    # Write manifest
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="constraint_verification",
        data_type="synthetic",
        explicit_model_source="known coefficients",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": N_TRAIN},
        train_test_split=None,
        random_seed=42,
        noise_level=0.5,
        constraints_used={c["name"]: c for c in CONSTRAINTS},
        metrics={"recommendation_safe": recommendation_safe},
        all_checks_pass=recommendation_safe,
        plot_files=[],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=["output/constraint_verification.csv"],
        duration_seconds=round(duration, 2),
    )
    print(f"  Manifest: {os.path.join(OUTPUT_DIR, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
