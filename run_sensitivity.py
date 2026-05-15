"""
Sample-size sensitivity test.

Measures how prediction accuracy degrades as the number of training
observations decreases.  Collects a large pool of observations from the
TCP server, then trains the modelless predictor on progressively smaller
subsets while evaluating on a fixed hold-out test set.

Outputs:
  - output/sample_size_sensitivity.csv
  - output/run_manifest.json
  - Console summary table

Requires a running plant server on 127.0.0.1:9100.
"""

import csv
import os
import time

import numpy as np
from sklearn.metrics import mean_squared_error

from src.data_collector import collect
from src.manifest import write_manifest
from src.modelless_predictor import ModellessPredictor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_SIZES = [5, 10, 25, 50, 100, 200]
TEST_SIZE = 100


def main():
    np.random.seed(42)
    t_start = time.time()

    # Collect a single large pool + fixed test set from the server
    max_train = max(SAMPLE_SIZES)
    print(f"Collecting {max_train} training + {TEST_SIZE} test observations …")
    pool_in, pool_out = collect(max_train)
    test_in, test_out = collect(TEST_SIZE)
    print(f"  Pool: {pool_in.shape}  Test: {test_in.shape}\n")

    # Run sensitivity sweep
    rows = []
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    print(f"  {'n':>5s}  {'yield_mae':>10s} {'yield_rmse':>11s} {'yield_r2':>10s}"
          f"  {'purity_mae':>11s} {'purity_rmse':>12s} {'purity_r2':>10s}")
    print("  " + "-" * 75)

    for n in SAMPLE_SIZES:
        train_in = pool_in[:n]
        train_out = pool_out[:n]

        predictor = ModellessPredictor()
        predictor.fit(train_in, train_out, output_names=["yield", "purity"])

        metrics = predictor.evaluate(test_in, test_out)
        preds = predictor.predict(test_in)

        # RMSE (not in evaluate() — compute manually)
        yield_rmse = float(np.sqrt(mean_squared_error(test_out[:, 0], preds[:, 0])))
        purity_rmse = float(np.sqrt(mean_squared_error(test_out[:, 1], preds[:, 1])))

        row = {
            "run_id": run_id,
            "sample_size": n,
            "yield_mae": round(metrics["yield"]["mae"], 6),
            "yield_rmse": round(yield_rmse, 6),
            "yield_r2": round(metrics["yield"]["r2"], 6),
            "purity_mae": round(metrics["purity"]["mae"], 6),
            "purity_rmse": round(purity_rmse, 6),
            "purity_r2": round(metrics["purity"]["r2"], 6),
        }
        rows.append(row)

        print(f"  {n:5d}  {row['yield_mae']:10.6f} {row['yield_rmse']:11.6f} {row['yield_r2']:10.6f}"
              f"  {row['purity_mae']:11.6f} {row['purity_rmse']:12.6f} {row['purity_r2']:10.6f}")

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "sample_size_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved to: {csv_path}")

    # Write manifest
    duration = time.time() - t_start
    manifest_path = write_manifest(
        OUTPUT_DIR,
        analysis_type="sample_size_sensitivity",
        data_type="synthetic",
        explicit_model_source="external TCP server (127.0.0.1:9100)",
        explicit_model_version=None,
        modelless_model_type="LinearRegression",
        sample_size={"pool": max_train, "test": TEST_SIZE},
        train_test_split={"sample_sizes": SAMPLE_SIZES, "test": TEST_SIZE},
        random_seed=42,
        noise_level=0.5,
        constraints_used=None,
        metrics={str(r["sample_size"]): {k: r[k] for k in r if k not in ("run_id", "sample_size")} for r in rows},
        all_checks_pass=None,
        plot_files=[],
        metric_files=["output/sample_size_sensitivity.csv"],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )
    print(f"  Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
