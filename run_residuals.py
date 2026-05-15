"""
Residual / error analysis.

Trains the modelless predictor, evaluates on a hold-out test set against
noise-free ground truth, and saves per-observation residuals plus summary
statistics.

Outputs:
  - output/residual_analysis.csv   (per-row residuals)
  - output/residual_summary.csv    (summary statistics)
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_TRAIN = 300
N_TEST = 200
INPUT_NAMES = ["temperature", "flow_rate", "concentration"]
OUTPUT_NAMES = ["yield", "purity"]


def main():
    np.random.seed(42)
    t_start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Collect data
    print(f"Collecting {N_TRAIN} training + {N_TEST} test observations …")
    train_in, train_out = collect(N_TRAIN)
    test_in, test_out = collect(N_TEST)

    # Train predictor
    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=OUTPUT_NAMES)
    preds = predictor.predict(test_in)

    # Residuals = actual (noisy) – predicted
    residuals = test_out - preds

    # ── Per-observation CSV ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "residual_analysis.csv")
    fieldnames = (
        ["run_id", "obs"]
        + [f"input_{name}" for name in INPUT_NAMES]
        + [f"actual_{name}" for name in OUTPUT_NAMES]
        + [f"predicted_{name}" for name in OUTPUT_NAMES]
        + [f"residual_{name}" for name in OUTPUT_NAMES]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j in range(len(test_in)):
            row = {"run_id": run_id, "obs": j}
            for k, name in enumerate(INPUT_NAMES):
                row[f"input_{name}"] = round(float(test_in[j, k]), 6)
            for k, name in enumerate(OUTPUT_NAMES):
                row[f"actual_{name}"] = round(float(test_out[j, k]), 6)
                row[f"predicted_{name}"] = round(float(preds[j, k]), 6)
                row[f"residual_{name}"] = round(float(residuals[j, k]), 6)
            writer.writerow(row)
    print(f"  Per-observation CSV: {csv_path}")

    # ── Summary statistics ───────────────────────────────────────────────
    summary_rows = []
    for k, name in enumerate(OUTPUT_NAMES):
        r = residuals[:, k]
        abs_r = np.abs(r)
        summary_rows.append({
            "output": name,
            "mean_residual": round(float(r.mean()), 6),
            "mean_abs_residual": round(float(abs_r.mean()), 6),
            "max_abs_residual": round(float(abs_r.max()), 6),
            "p50_abs_error": round(float(np.percentile(abs_r, 50)), 6),
            "p90_abs_error": round(float(np.percentile(abs_r, 90)), 6),
            "p95_abs_error": round(float(np.percentile(abs_r, 95)), 6),
        })

    summary_path = os.path.join(OUTPUT_DIR, "residual_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Summary CSV:        {summary_path}")

    # Print summary table
    print()
    print(f"  {'output':>8s}  {'mean':>10s}  {'|mean|':>10s}  {'max|r|':>10s}"
          f"  {'P50':>10s}  {'P90':>10s}  {'P95':>10s}")
    print("  " + "-" * 70)
    for s in summary_rows:
        print(f"  {s['output']:>8s}  {s['mean_residual']:10.6f}  {s['mean_abs_residual']:10.6f}"
              f"  {s['max_abs_residual']:10.6f}  {s['p50_abs_error']:10.6f}"
              f"  {s['p90_abs_error']:10.6f}  {s['p95_abs_error']:10.6f}")

    # Write manifest
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="residual_analysis",
        data_type="synthetic",
        explicit_model_source="external TCP server (127.0.0.1:9100)",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"train": N_TRAIN, "test": N_TEST},
        train_test_split={"train": N_TRAIN, "test": N_TEST},
        random_seed=42,
        noise_level=0.5,
        constraints_used=None,
        metrics={s["output"]: {k: s[k] for k in s if k != "output"} for s in summary_rows},
        all_checks_pass=None,
        plot_files=[],
        metric_files=["output/residual_summary.csv"],
        prediction_files=[],
        residual_files=["output/residual_analysis.csv"],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )
    print(f"\n  Manifest: {os.path.join(OUTPUT_DIR, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
