"""
Prediction/Inference functions for retail sales forecasting.
Load trained models and generate forecasts.
"""
import mlflow
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src.utils import (
    create_output_dir,
    get_current_timestamp,
    load_config,
    get_model_config,
    get_config_value,
    get_mlflow_tracking_uri
)

# Load config once
_CONFIG = load_config()
FORECAST_HORIZON = get_config_value(_CONFIG, 'forecast', 'horizon', default=7)
FORECAST_FREQ    = get_config_value(_CONFIG, 'forecast', 'frequency', default='D')
EVAL_METRIC      = get_config_value(_CONFIG, 'model', 'eval_metric', default='MASE')
MLFLOW_EXPERIMENT = get_config_value(
    _CONFIG, 'mlflow', 'experiment_name', default='rossmann-forecasting'
)

# LOAD MODEL

def load_model(model_path):
    """
    Load a trained TimeSeriesPredictor model.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        Loaded TimeSeriesPredictor
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from: {model_path}")
    
    # Check if path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load the predictor
    predictor = TimeSeriesPredictor.load(model_path)
    logger.info("Model loaded successfully!")
    logger.info(f"Prediction length: {predictor.prediction_length}")
    logger.info(f"Target column: {predictor.target}")
    
    return predictor


# MAKE PREDICTIONS

def make_predictions(predictor, data):
    """
    Generate forecasts using the trained model.
    
    Args:
        predictor: Trained TimeSeriesPredictor
        data: Input data (pandas DataFrame or TimeSeriesDataFrame)
    
    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Generating predictions")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(data, TimeSeriesDataFrame):
        from src.train import convert_to_timeseries_dataframe
        data = convert_to_timeseries_dataframe(data, freq=predictor.freq)
    
    logger.info(f"Predicting for {len(data.item_ids)} stores/items")
    
    # Make predictions
    # Quantiles are already configured in the trained model
    predictions = predictor.predict(data)
    
    logger.info("Predictions generated successfully!")
    logger.info(f"Prediction shape: {predictions.shape}")
    
    return predictions


# FORMAT PREDICTIONS

def format_predictions_for_export(predictions):
    """
    Convert predictions to a clean pandas DataFrame for export.
    
    Args:
        predictions: TimeSeriesDataFrame with predictions
    
    Returns:
        Clean pandas DataFrame with columns: store_id, date, predicted_sales, lower_bound, upper_bound
    """
    logger = logging.getLogger(__name__)
    logger.info("Formatting predictions for export...")
    
    # Convert to regular pandas DataFrame
    pred_list = []
    
    for item_id in predictions.item_ids:
        item_preds = predictions.loc[item_id]
        
        # Create a dataframe for this store
        store_df = pd.DataFrame({
            'store_id': item_id,
            'date': item_preds.index,
            'predicted_sales': item_preds['mean'].values if 'mean' in item_preds.columns else item_preds.iloc[:, 0].values
        })
        
        # Add quantiles if available
        if '0.1' in item_preds.columns:
            store_df['lower_bound'] = item_preds['0.1'].values
        if '0.9' in item_preds.columns:
            store_df['upper_bound'] = item_preds['0.9'].values
        
        pred_list.append(store_df)
    
    # Combine all stores
    final_df = pd.concat(pred_list, ignore_index=True)
    
    # Sort by store and date
    final_df = final_df.sort_values(['store_id', 'date']).reset_index(drop=True)
    
    logger.info(f"Formatted predictions: {final_df.shape}")
    
    return final_df


# SAVE PREDICTIONS

def save_predictions(predictions_df, filename=None, output_dir="outputs"):
    """
    Save predictions to CSV file.
    
    Args:
        predictions_df: DataFrame with predictions
        filename: Name of the output file (optional, auto-generated if None)
        output_dir: Output directory name
    
    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    out_dir = create_output_dir(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = get_current_timestamp()
        filename = f"predictions_{timestamp}.csv"
    
    # Full path
    filepath = Path(out_dir) / filename
    
    # Save to CSV
    predictions_df.to_csv(filepath, index=False)
    logger.info(f"Predictions saved to: {filepath}")
    
    return str(filepath)


# PREDICTION SUMMARY

def generate_prediction_summary(predictions_df):
    """
    Generate summary statistics for predictions.
    
    Args:
        predictions_df: DataFrame with predictions
    
    Returns:
        Dictionary with summary statistics
    """
    logger = logging.getLogger(__name__)
    
    summary = {
        'num_stores': predictions_df['store_id'].nunique(),
        'num_predictions': len(predictions_df),
        'date_range': {
            'start': str(predictions_df['date'].min()),
            'end': str(predictions_df['date'].max())
        },
        'predicted_sales': {
            'total': float(predictions_df['predicted_sales'].sum()),
            'mean': float(predictions_df['predicted_sales'].mean()),
            'median': float(predictions_df['predicted_sales'].median()),
            'min': float(predictions_df['predicted_sales'].min()),
            'max': float(predictions_df['predicted_sales'].max())
        }
    }
    
    logger.info("\n" + "="*60)
    logger.info("Prediction Summary")
    logger.info("="*60)
    logger.info(f"Number of stores: {summary['num_stores']}")
    logger.info(f"Total predictions: {summary['num_predictions']}")
    logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"Total predicted sales: ${summary['predicted_sales']['total']:,.2f}")
    logger.info(f"Average predicted sales per day: ${summary['predicted_sales']['mean']:,.2f}")
    logger.info("="*60)
    
    return summary


# FULL PREDICTION PIPELINE

def predict_pipeline(model_path, data, save_output=True, zero_shot=False, config=None):
    """
    Full prediction pipeline.
    Load model, make predictions, format, and save.
    
    Args:
        model_path: Path to trained model OR zero-shot model name
        data: Input data for prediction
        save_output: Whether to save predictions to CSV
        zero_shot: Whether this is zero-shot mode (bool or 'yes'/'no')
    
    Returns:
        Tuple of (predictions_df, summary, save_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Use global config if none provided
    if config is None:
        config = _CONFIG

    # Normalize zero_shot flag to a boolean
    zero_shot_flag = str(zero_shot).lower() in ("yes", "true", "1")

    # Make sure model_path is a string
    model_path_str = str(model_path)

    # Setup MLflow for zero-shot mode
    if zero_shot_flag:
        tracking_uri = get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        # Start MLflow run for zero-shot
        mlflow.start_run(run_name=f"zeroshot_{get_current_timestamp()}")

        # Log zero-shot parameters
        mlflow.log_param("mode", "zero_shot")
        mlflow.log_param("zero_shot_model", model_path_str)
        mlflow.log_param("training_time", 0)
        mlflow.log_param("training", "skipped")

    # Step 1: Load model or use zero-shot
    if zero_shot_flag:
        # Zero-shot mode: use pre-trained model directly
        logger.info(f"Using ZERO-SHOT mode with model: {model_path_str}")
        logger.info("Zero-shot mode: Creating new Chronos predictor")

        # Use FULL config (passed in or from config.yaml)
        model_config = get_model_config(config)

        # Convert data if needed
        if not isinstance(data, TimeSeriesDataFrame):
            from src.train import convert_to_timeseries_dataframe
            data = convert_to_timeseries_dataframe(data, freq=model_config["freq"])

        # Create zero-shot predictor using Chronos2
        predictor = TimeSeriesPredictor(
            prediction_length=model_config["prediction_length"],
            target="target",
            freq=model_config["freq"],
            eval_metric=model_config["eval_metric"],
        ).fit(
            train_data=data,
            presets=config.get("model", {}).get("chronos_preset", "chronos2_small"),
            skip_model_selection=True,  # Skip model selection for zero-shot
        )
    else:
        # Normal mode: load saved model from disk
        logger.info(f"Loading model from: {model_path_str}")
        predictor = load_model(model_path_str)

    # Step 2: Make predictions
    predictions = make_predictions(predictor, data)

    # Step 3: Format predictions
    predictions_df = format_predictions_for_export(predictions)

    # Step 4: Generate summary
    summary = generate_prediction_summary(predictions_df)

    # Step 5: Save predictions
    save_path = None
    if save_output:
        save_path = save_predictions(predictions_df)

    # Log to MLflow if zero-shot
    if zero_shot_flag:
        # Log metrics
        mlflow.log_metric("num_predictions", len(predictions_df))
        mlflow.log_metric("num_stores", predictions_df["store_id"].nunique())
        mlflow.log_metric("total_predicted_sales", float(summary["predicted_sales"]["total"]))
        mlflow.log_metric("avg_predicted_sales", float(summary["predicted_sales"]["mean"]))

        # Log the predictions file (only if we actually saved one)
        if save_path is not None:
            mlflow.log_artifact(save_path, artifact_path="predictions")

        # End MLflow run
        mlflow.end_run()
        logger.info("   Zero-shot run logged to MLflow")

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return predictions_df, summary, save_path



# HELPER: PREDICT FUTURE (NO HISTORICAL DATA)

def predict_future(model_path, train_data, num_steps=FORECAST_HORIZON):
    """
    Predict future values beyond the training data.
    Useful for forecasting into the future when you don't have recent data.
    
    Args:
        model_path: Path to trained model
        train_data: Historical training data
        num_steps: Number of steps to forecast into future
    
    Returns:
        DataFrame with future predictions
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Forecasting {num_steps} steps into the future...")
    
    # Load model
    predictor = load_model(model_path)
    
    # Convert to TimeSeriesDataFrame
    if not isinstance(train_data, TimeSeriesDataFrame):
        from src.train import convert_to_timeseries_dataframe
        train_data = convert_to_timeseries_dataframe(train_data, freq=predictor.freq)
    
    # Make predictions (AutoGluon will automatically forecast future)
    predictions = predictor.predict(train_data)
    
    # Format and return
    predictions_df = format_predictions_for_export(predictions)
    
    logger.info(f"Future forecast complete! {len(predictions_df)} predictions generated.")
    
    return predictions_df


# HELPER: BATCH PREDICTION

def batch_predict_by_store(model_path, data, store_ids):
    """
    Make predictions for specific stores only.
    Useful for testing or selective forecasting.
    
    Args:
        model_path: Path to trained model
        data: Input data containing all stores
        store_ids: List of store IDs to predict for
    
    Returns:
        DataFrame with predictions for selected stores
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Making predictions for {len(store_ids)} specific stores...")
    
    # Filter data to selected stores
    store_ids = [str(s) for s in store_ids]  # Convert to strings
    filtered_data = data[data['item_id'].isin(store_ids)].copy()
    
    logger.info(f"Filtered data to {len(filtered_data)} rows")
    
    # Run prediction pipeline
    predictions_df, summary, save_path = predict_pipeline(
        model_path=model_path,
        data=filtered_data,
        save_output=True
    )
    
    return predictions_df