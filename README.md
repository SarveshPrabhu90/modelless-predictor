# Modelless Predictor — Data-Driven Process Prediction

A data-driven predictor that learns to predict pharmaceutical process outputs
by observing input-output pairs streamed over TCP/IP — **without any knowledge
of the underlying model coefficients**.

## Core Idea

Instead of relying on a hand-crafted explicit model, this project:

1. **Collects** input-output observations from a plant server via TCP/IP
2. **Learns** the system's behavior using linear regression
3. **Predicts** future outputs with near-identical accuracy (R² > 0.999)
4. **Optimizes** control inputs via linear programming to maximize yield
   subject to a purity constraint

## Architecture

```
┌──────────────────────┐         TCP/IP          ┌──────────────────────┐
│   Plant Server       │ ◄──────────────────────► │   Data Collector     │
│   (external)         │   JSON over sockets      │   data_collector.py  │
└──────────────────────┘                          └─────────┬────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │  Modelless Predictor │
                                                  │  (Linear Regression) │
                                                  │  modelless_predictor │
                                                  └─────────┬────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │     LP Optimizer     │
                                                  │  (scipy.linprog)     │
                                                  │  optimizer.py        │
                                                  └──────────────────────┘
```

## Key Technologies

| Area         | Tool / Technique              |
|--------------|-------------------------------|
| Language     | Python 3.10+                  |
| Networking   | TCP/IP sockets, JSON protocol |
| Learning     | scikit-learn LinearRegression |
| Optimization | scipy linprog (HiGHS)         |

## Quick Start

Requires a running plant server (e.g., from [explicit-model](https://github.com/SarveshPrabhu90/explicit-model)) on `127.0.0.1:9100`.

```bash
pip install -r requirements.txt
python run_predictor.py
```

## Project Structure

```
modelless-predictor/
├── README.md
├── requirements.txt
├── run_predictor.py              # Entry point: collect, learn, optimize
├── src/
│   ├── __init__.py
│   ├── data_collector.py         # TCP client gathering observations
│   ├── modelless_predictor.py    # Linear regression learner
│   └── optimizer.py              # LP optimization using learned model
└── tests/
    ├── __init__.py
    ├── test_data_collector.py
    ├── test_modelless_predictor.py
    └── test_optimizer.py
```

## Sample-Size Sensitivity

Tests how many observations the predictor needs before it becomes useful:

```bash
python run_sensitivity.py
```

Sweeps over sample sizes [5, 10, 25, 50, 100, 200], trains on each subset,
evaluates against a fixed 100-sample test set, and reports MAE, RMSE, and R²
for both yield and purity. Results are saved to `output/sample_size_sensitivity.csv`
with a `run_manifest.json`.

**How to interpret:** R² should climb toward 1.0 as n increases, with diminishing
returns beyond a certain point. That point is the practical minimum sample size.

## Run Manifest

Both `python run_predictor.py` and `python validate.py` save a
`run_manifest.json` to the `output/` folder. The manifest records run metadata
(timestamp, sample sizes, train/test split, learned coefficients, metrics,
optimization results) for every run. See `docs/SHARED_OUTPUT_CONTRACT.md` in the
workspace root for the full schema.

## Next Steps

- See [GitHub Issues](../../issues) for planned work.
