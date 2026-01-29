# Time Series Forecasting MLOps Pipeline  
**Luigi · AutoGluon · Chronos · MLflow · Config-Driven**

This repository implements a **production-ready, config-driven time series forecasting pipeline** designed for enterprise use cases. It supports both **traditional model training** and **zero-shot forecasting** using foundation models, with full experiment tracking and deterministic orchestration.

Blog: [Forecasting at Scale With Foundation Models: A Practical Guide](https://medium.com/@rameshnikkitha/forecasting-at-scale-with-foundation-models-a-complete-guide-891ed8cdd056)

---

## Key Features

- **Luigi** for clear, deterministic pipeline orchestration  
- **AutoGluon TimeSeries** for traditional training and evaluation  
- **Chronos-2** foundation models for zero-shot forecasting  
- **MLflow** for experiment tracking and comparison  
- **YAML-based configuration** as the single source of truth  
- **Platform-agnostic** (local, cloud, CI, on-prem)  

---

## Project Structure

```
timeseries-forecasting-mlops/
├── config.yaml
├── run.py
├── data/
│   ├── train.csv
│   └── store.csv
├── src/
│   ├── pipeline.py      # Luigi tasks
│   ├── preprocess.py    # Data preprocessing
│   ├── train.py         # Training logic
│   ├── predict.py       # Inference logic
│   └── utils.py         # Shared utilities
├── outputs/             # Forecast outputs
├── models/              # Saved models (traditional mode)
├── luigi_outputs/       # Luigi task artifacts
└── mlruns/              # MLflow tracking store
```

---

## Data Source

This pipeline uses the **Rossmann Store Sales** dataset from Kaggle:

**Dataset:** [https://www.kaggle.com/c/rossmann-store-sales](https://www.kaggle.com/c/rossmann-store-sales)

**Required files:**
* `train.csv` – historical sales data
* `store.csv` – store metadata

**Setup:**
1. Download the dataset from Kaggle
2. Place `train.csv` and `store.csv` in the `data/` directory
3. The pipeline will automatically preprocess and merge these files

---

## 1. Environment Setup

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Recommended Python version:** `3.11`

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Running the Pipeline

Once dependencies are installed and `config.yaml` is configured:

```bash
python3.11 run.py
```

This triggers the full Luigi DAG:

```
PreprocessTask → TrainTask (optional) → PredictTask
```

---

## 4. Configuration Overview (`config.yaml`)

All pipeline behavior is controlled via `config.yaml`.

### Core Sections

* `data` – column mappings
* `forecast` – frequency, horizon, validation split
* `model` – zero-shot vs traditional behavior
* `training` – time limits and presets (traditional only)
* `pipeline.mode` – quick / medium / production
* `mlflow` – experiment tracking
* `output` – artifact directories

---

## 5. Pipeline Modes (Quick / Medium / Production)

Set the execution profile via:

```yaml
pipeline:
  mode: production   # quick, medium, production
```

### What modes control

Modes apply overrides defined under `modes:`:

* **quick** – fast iteration, limited data
* **medium** – balanced experimentation
* **production** – full data, higher compute budget

Example:

```yaml
modes:
  quick:
    training:
      num_stores: 3
      time_limit: 120
      presets: fast_training
```

To switch modes:

```yaml
pipeline:
  mode: quick
```

Then run:

```bash
python3.11 run.py
```

---

## 6. Zero-Shot vs Traditional Forecasting

### Zero-Shot Forecasting (Chronos-2)

Use zero-shot mode when you want:

* No model training
* Minimal compute cost
* Low maintenance overhead
* Strong baseline performance at scale

**Config:**

```yaml
model:
  mode: chronos_zero_shot
  zero_shot: "yes"
  chronos_preset: chronos2_small
  zero_shot_model: chronos2_small
```

**Behavior:**

* `TrainTask` is skipped
* Predictions generated directly using Chronos-2
* No model artifacts saved

Run:

```bash
python3.11 run.py
```

---

### Traditional Training (AutoGluon)

Use traditional mode when you need:

* Maximum accuracy
* Metric-aligned optimization
* Persisted models for reuse

**Config:**

```yaml
model:
  mode: traditional
  zero_shot: "no"
```

**Behavior:**

* Full training via AutoGluon
* Model saved to `models/`
* Loaded during prediction

Run:

```bash
python3.11 run.py
```

---

## 7. Evaluation Metric Selection

Set the optimization metric via:

```yaml
model:
  eval_metric: MAPE   # RMSE, MAPE, MASE, MAE
```

**Guidance:**

* **MAPE** – business-facing forecasts (sales, demand)
* **RMSE** – penalizes large errors
* **MASE** – stable cross-series comparison

---

## 8. Forecast Horizon & Validation Window

```yaml
forecast:
  frequency: D
  horizon: 3
  validation_days: 3
```

* `horizon` – number of future periods to forecast
* `validation_days` – holdout window for evaluation

---

## 9. Outputs & Artifacts

Default locations:

* Predictions → `outputs/`
* Models → `models/` (traditional mode only)
* Luigi artifacts → `luigi_outputs/`
* MLflow runs → `mlruns/`

Configurable via:

```yaml
output:
  predictions_dir: outputs
  models_dir: models
  luigi_dir: luigi_outputs
```

---

## 10. MLflow Experiment Tracking

### Viewing the MLflow UI

To view experiment metrics, parameters, and compare runs:

```bash
mlflow ui
```

This starts the MLflow tracking server at `http://localhost:5000`

Open your browser and navigate to:

```
http://localhost:5000
```

### What you can track

* **Metrics** – MAPE, RMSE, MASE, MAE across validation windows
* **Parameters** – model mode, forecast horizon, evaluation metric
* **Artifacts** – predictions, model metadata
* **Run comparison** – side-by-side evaluation of zero-shot vs traditional

### Custom port

To run on a different port:

```bash
mlflow ui --port 8080
```

### MLflow configuration

Configure experiment name in `config.yaml`:

```yaml
mlflow:
  experiment_name: timeseries_forecasting
  tracking_uri: ./mlruns
```

---

## 11. Example Configurations

### Production + Zero-Shot

```yaml
pipeline:
  mode: production
model:
  mode: chronos_zero_shot
  zero_shot: "yes"
  chronos_preset: chronos2_small
  eval_metric: MAPE
```

### Medium + Traditional

```yaml
pipeline:
  mode: medium
model:
  mode: traditional
  zero_shot: "no"
  eval_metric: RMSE
```

---

## Troubleshooting

### `KeyError: 'pipeline'`

Ensure `config.yaml` contains:

```yaml
pipeline:
  mode: production
```

### Identical predictions for zero-shot and traditional

Verify that:

* Zero-shot uses:

  ```yaml
  mode: chronos_zero_shot
  zero_shot: "yes"
  ```
* Traditional uses:

  ```yaml
  mode: traditional
  zero_shot: "no"
  ```

---

## Summary

This pipeline is designed to let teams:

* Default to **zero-shot forecasting** for scale and cost efficiency
* Selectively apply **traditional training** where accuracy matters most
* Operate entirely via configuration, not code changes

A flexible, enterprise-ready forecasting system — built for 2026 and beyond.
