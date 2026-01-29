"""
Training functions for retail sales forecasting with AutoGluon Chronos.
Includes MLflow experiment tracking.
"""
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.utils import (
    create_output_dir,
    get_config_value,
    get_current_timestamp,
    get_mlflow_tracking_uri,
    get_model_config,
    load_config,
    log_dict_as_params,
    log_metrics_dict,
)


# Load config once at import time
_CONFIG = load_config()

# Global defaults from config.yaml
FORECAST_HORIZON = get_config_value(_CONFIG, 'forecast', 'horizon', default=7)
FORECAST_FREQ    = get_config_value(_CONFIG, 'forecast', 'frequency', default='D')
EVAL_METRIC      = get_config_value(_CONFIG, 'model', 'eval_metric', default='MASE')
MLFLOW_EXPERIMENT = get_config_value(
    _CONFIG, 'mlflow', 'experiment_name', default='rossmann-forecasting'
)

TRAINING_TIME_LIMIT = get_config_value(
    _CONFIG, "training", "time_limit", default=600
)
TRAINING_PRESETS = get_config_value(
    _CONFIG, "training", "presets", default="medium_quality"
)

# MLFLOW SETUP

def setup_mlflow(experiment_name=None):
    """
    Initialize MLflow tracking.
    """
    logger = logging.getLogger(__name__)

    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT
    
    # Set tracking URI to local directory
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
    
    return experiment.experiment_id


# DATA CONVERSION FOR AUTOGLUON

def convert_to_timeseries_dataframe(df, freq='D'):
    """
    Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame.
    Handles missing dates by filling them.
    
    Args:
        df: DataFrame with timestamp, target, item_id columns
        freq: Frequency of the time series ('D' for daily, 'W' for weekly, etc.)
    
    Returns:
        TimeSeriesDataFrame for AutoGluon
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting to TimeSeriesDataFrame...")
    
    # AutoGluon expects specific column names
    # timestamp -> index, item_id stays, target stays
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    logger.info(f"Created TimeSeriesDataFrame with {len(ts_df.item_ids)} items")
    
    # Convert to regular frequency (fill missing dates)
    # This is CRITICAL for AutoGluon to work properly
    logger.info(f"Converting to regular frequency: {freq}")
    ts_df = ts_df.convert_frequency(freq=freq)
    logger.info("Frequency conversion complete!")
    
    return ts_df


# MODEL TRAINING

def train_model(
        train_df,
        prediction_length=FORECAST_HORIZON,
        time_limit=TRAINING_TIME_LIMIT,
        model_path=None,
        freq=FORECAST_FREQ):
    """
    Train AutoGluon TimeSeriesPredictor.
    
    Args:
        train_df: Training data (pandas DataFrame or TimeSeriesDataFrame)
        prediction_length: Number of days to forecast (from config)
        time_limit: Training time limit in seconds (default: 600 = 10 mins)
        model_path: Path to save the model (optional)
        freq: Frequency of time series ('D' = daily, 'W' = weekly, etc.)
    
    Returns:
        Trained TimeSeriesPredictor
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Starting model training")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(train_df, TimeSeriesDataFrame):
        train_df = convert_to_timeseries_dataframe(train_df, freq=freq)
    
    # Create model save path if not provided
    if prediction_length is None:
        raise ValueError("prediction_length must be provided from config")
    
    if model_path is None:
        model_dir = create_output_dir("models")
        timestamp = get_current_timestamp()
        model_path = f"{model_dir}/model_{timestamp}"
    
    logger.info(f"Model will be saved to: {model_path}")
    logger.info(f"Prediction length: {prediction_length} days")
    logger.info(f"Time limit: {time_limit} seconds")
    logger.info(f"Frequency: {freq}")
    logger.info(f"Eval metric: {EVAL_METRIC}")
    logger.info(f"Training presets: {TRAINING_PRESETS}")
    
    # Initialize the predictor with frequency specified
    predictor = TimeSeriesPredictor(
        path=model_path,
        target='target',  # Column we're forecasting
        prediction_length=prediction_length,
        eval_metric=EVAL_METRIC,
        freq=freq, 
        verbosity=2  
    )
    
    # Train the model
    # AutoGluon will automatically try multiple models and ensemble them
    logger.info("Training started...")
    predictor.fit(
        train_data=train_df,
        time_limit=time_limit,
        presets=TRAINING_PRESETS,  
        skip_model_selection=False  
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_path}")
    
    return predictor


# MODEL EVALUATION

def evaluate_model(predictor, val_df, freq=FORECAST_FREQ):
    """
    Evaluate the trained model on validation data.
    
    Args:
        predictor: Trained TimeSeriesPredictor
        val_df: Validation data (pandas DataFrame or TimeSeriesDataFrame)
        freq: Frequency of time series (from config.yaml)
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Evaluating model on validation set")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(val_df, TimeSeriesDataFrame):
        val_df = convert_to_timeseries_dataframe(val_df, freq=freq)
    
    # Get predictions on validation set
    logger.info("Generating predictions...")
    predictions = predictor.predict(val_df)

    metrics = {}
    
    # Calculate metrics using AutoGluon's built-in evaluation
    try:
        # Try to get leaderboard with validation scores
        leaderboard = predictor.leaderboard(val_df, silent=True)
        best_model = leaderboard.iloc[0]
        
        metrics.update(
            {
                'val_MASE': float(best_model['score_val']),
                'best_model': str(best_model['model']),
                'num_models_trained': len(leaderboard)
            }
        )
    except Exception as e:
        logger.warning(f"Could not get leaderboard: {e}")
        metrics.update(
            {
                'best_model': 'unknown',
                'num_models_trained': 0
            }
        )
    
    # Calculate additional metrics manually
    # Get actual values and predictions
    try:
        # Align predictions with validation data
        actual_list = []
        pred_list = []
        
        for item_id in val_df.item_ids:
            val_item = val_df.loc[item_id]
            pred_item = predictions.loc[item_id]
            
            # Get overlapping timestamps
            common_timestamps = val_item.index.intersection(pred_item.index)
            
            if len(common_timestamps) > 0:
                actual_list.extend(val_item.loc[common_timestamps, 'target'].values)
                pred_list.extend(pred_item.loc[common_timestamps, 'mean'].values)
        
        if len(actual_list) > 0:
            actuals = np.array(actual_list)
            preds = np.array(pred_list)
            
            # Calculate metrics
            mae = np.mean(np.abs(actuals - preds))
            rmse = np.sqrt(np.mean((actuals - preds) ** 2))
            
            # MAPE (avoid division by zero)
            mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))) * 100
            
            metrics['val_MAE'] = float(mae)
            metrics['val_RMSE'] = float(rmse)
            metrics['val_MAPE'] = float(mape)
    
    except Exception as e:
        logger.warning(f"Could not calculate additional metrics: {e}")
    
    logger.info("Evaluation metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    return metrics


# MLFLOW LOGGING

def log_training_to_mlflow(predictor, metrics, config):
    """
    Log training parameters, metrics, and model to MLflow.
    
    Args:
        predictor: Trained predictor
        metrics: Dictionary of evaluation metrics
        config: Dictionary of training configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Logging to MLflow...")
    
    # Separate metrics into numeric (for mlflow.log_metric) and string (for mlflow.log_param)
    numeric_metrics = {}
    string_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric_metrics[key] = value
        else:
            string_metrics[key] = value
    
    # Log parameters (config + string metrics)
    log_dict_as_params(config, prefix="config_")
    log_dict_as_params(string_metrics, prefix="")
    
    # Log only numeric metrics
    log_metrics_dict(numeric_metrics)
    
    # Log model leaderboard as artifact
    try:
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard_path = f"{predictor.path}/leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path, artifact_path="model_info")
    except Exception as e:
        logger.warning(f"Could not log leaderboard: {e}")
    
    # Log the model path
    mlflow.log_param("model_path", predictor.path)
    
    logger.info("MLflow logging complete!")


# ============================================================================
# FULL TRAINING PIPELINE
# ============================================================================

def train_pipeline(train_df, val_df, config=None):
    """
    Full training pipeline with MLflow tracking.
    Supports Chronos zero-shot, fine-tuning, and traditional training.
    
    Args:
        train_df: Training data
        val_df: Validation data
        config: Training configuration dict
    
    Returns:
        Tuple of (predictor, metrics, model_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load global config if not provided
    if config is None:
        config = _CONFIG

    # Parse model configuration
    model_config = get_model_config(config)
    mode = model_config["mode"]
    logger.info(f"Model mode: {mode}")

    # Training settings
    training_cfg = config.get("training", {}) or {}
    time_limit = training_cfg.get("time_limit", TRAINING_TIME_LIMIT)

    # Setup MLflow
    experiment_id = setup_mlflow()
    logger.info(f"Using MLflow experiment ID: {experiment_id}")

    # Start MLflow run
    with mlflow.start_run(run_name=f"{mode}_{get_current_timestamp()}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Convert training data if needed
        if not isinstance(train_df, TimeSeriesDataFrame):
            train_df = convert_to_timeseries_dataframe(train_df, freq=model_config["freq"])

        # ============================================================
        # ZERO-SHOT MODE (NO MODEL SAVED, NO TRAINING TASK)
        # ============================================================
        if mode == "chronos_zero_shot":
            logger.info("Using CHRONOS-2 ZERO-SHOT mode (no training)")

            predictor = TimeSeriesPredictor(
                prediction_length=model_config["prediction_length"],
                target="target",
                freq=model_config["freq"],
                eval_metric=model_config["eval_metric"],
            ).fit(
                train_data=train_df,
                tuning_data=train_df,
                hyperparameters={"Chronos2": {}},
                skip_model_selection=True
            )

            # Symbolic identifier (not a path)
            model_path = f"chronos_zero_shot_{model_config['chronos_preset']}"

        # ============================================================
        # FINE-TUNE CHRONOS-2 ON LOCAL DATA
        # ============================================================
        elif mode == "chronos_finetune":
            logger.info("Using CHRONOS-2 FINE-TUNING mode")

            predictor = TimeSeriesPredictor(
                prediction_length=model_config["prediction_length"],
                target="target",
                freq=model_config["freq"],
                eval_metric=model_config["eval_metric"],
            ).fit(
                train_data=train_df,
                hyperparameters={"Chronos2": {"fine_tune": True}},
                time_limit=time_limit
            )

            model_path = f"chronos_finetune_{model_config['chronos_preset']}"

        # ============================================================
        # TRADITIONAL TRAINING MODE (AUTOGLUON PRESETS)
        # ============================================================
        else:
            logger.info("Using TRADITIONAL TRAINING mode")

            predictor = train_model(
                train_df=train_df,
                prediction_length=model_config["prediction_length"],
                time_limit=time_limit,
                freq=model_config["freq"],
            )

            # train_model **saves a real model** and returns its saved path
            model_path = predictor.path

        # ============================================================
        # EVAL + LOGGING
        # ============================================================
        metrics = evaluate_model(predictor, val_df, freq=model_config["freq"])
        metrics["mode"] = mode

        log_training_to_mlflow(predictor, metrics, model_config)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Model: {model_path}")
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info("=" * 60)

    return predictor, metrics, model_path



# HELPER: QUICK TRAINING FOR TESTING

def quick_train(train_df, val_df, time_limit=300, config=None):
    """
    Quick training function for fast experimentation.
    Uses shorter time limit and fast_training preset.
    
    Args:
        train_df: Training data
        val_df: Validation data
        time_limit: Time limit in seconds (default: 300 = 5 mins)
        config: Configuration dict containing forecast horizon
    
    Returns:
        Tuple of (predictor, metrics, model_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("Running QUICK TRAINING (fast mode)...")
    
    if config is None:
        config = load_config()

    # Ensure training section exists
    config.setdefault("training", {})
    config["training"]["time_limit"] = time_limit
    config["training"]["presets"] = "fast_training"

    # prediction_length / freq / eval_metric are still taken from config.yaml
    return train_pipeline(train_df, val_df, config)
