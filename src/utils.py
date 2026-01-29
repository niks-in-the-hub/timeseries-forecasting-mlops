"""
Utility functions for the retail forecasting pipeline.
Simple helper functions - no classes, just functions!
"""

import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level="INFO"):
    """
    Set up basic logging configuration.
    
    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
    
    Returns:
        Logger object
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Get the directory where utils.py is located (src/)
    current_file = Path(__file__)
    # Go up one level to get project root
    project_root = current_file.parent.parent
    return project_root


def get_data_path(filename):
    """
    Get full path to a data file.
    
    Args:
        filename: Name of the file in data/ folder
    
    Returns:
        Full path to the file
    """
    root = get_project_root()
    data_path = root / "data" / filename
    return str(data_path)


def create_output_dir(dir_name="outputs"):
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_name: Name of the output directory
    
    Returns:
        Path to output directory
    """
    root = get_project_root()
    output_path = root / dir_name
    output_path.mkdir(exist_ok=True)
    return str(output_path)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_data():
    """
    Load the training data (train.csv).
    
    Returns:
        DataFrame with training data
    """
    logger = logging.getLogger(__name__)
    train_path = get_data_path("train.csv")
    
    logger.info(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(df)} rows of training data")
    
    return df


def load_store_data():
    """
    Load the store metadata (store.csv).
    
    Returns:
        DataFrame with store information
    """
    logger = logging.getLogger(__name__)
    store_path = get_data_path("store.csv")
    
    logger.info(f"Loading store data from {store_path}")
    df = pd.read_csv(store_path)
    logger.info(f"Loaded {len(df)} stores")
    
    return df


# ============================================================================
# MLFLOW HELPERS
# ============================================================================

def get_mlflow_tracking_uri():
    """
    Get MLflow tracking URI (local directory for now).
    
    Returns:
        MLflow tracking URI
    """
    root = get_project_root()
    mlruns_path = root / "mlruns"
    mlruns_path.mkdir(exist_ok=True)
    return str(mlruns_path)


def log_dict_as_params(params_dict, prefix=""):
    """
    Log a dictionary as MLflow parameters.
    Helper to flatten and log nested configs.
    
    Args:
        params_dict: Dictionary of parameters to log
        prefix: Optional prefix for parameter names
    """
    import mlflow
    
    for key, value in params_dict.items():
        param_name = f"{prefix}{key}" if prefix else key
        # MLflow params must be strings
        mlflow.log_param(param_name, str(value))


def log_metrics_dict(metrics_dict):
    """
    Log multiple metrics to MLflow at once.
    
    Args:
        metrics_dict: Dictionary of metric_name: value pairs
    """
    import mlflow
    
    for metric_name, value in metrics_dict.items():
        mlflow.log_metric(metric_name, value)


# ============================================================================
# DATE HELPERS
# ============================================================================

def get_date_range(df, date_column="Date"):
    """
    Get the min and max dates from a dataframe.
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column
    
    Returns:
        Tuple of (min_date, max_date)
    """
    df[date_column] = pd.to_datetime(df[date_column])
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    return min_date, max_date


def get_current_timestamp():
    """
    Get current timestamp as string for file naming.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# DATA VALIDATION
# ============================================================================

def check_file_exists(filepath):
    """
    Check if a file exists and raise error if not.
    
    Args:
        filepath: Path to check
    
    Raises:
        FileNotFoundError if file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required file not found: {filepath}")


def validate_columns(df, required_columns):
    """
    Check if DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Raises:
        ValueError if any required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


# ============================================================================
# SIMPLE STATS
# ============================================================================

def print_data_summary(df, name="Data"):
    """
    Print basic summary statistics for a dataframe.
    Simple helper for debugging.
    
    Args:
        df: DataFrame to summarize
        name: Name to display in output
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"{'='*50}\n")

# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config.yaml in project root)
    
    Returns:
        Dictionary with configuration
    """
    import yaml
    
    logger = logging.getLogger(__name__)
    
    # Get absolute path
    if not os.path.isabs(config_path):
        project_root = get_project_root()
        config_path = project_root / config_path
    
    # Check if config exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply mode-specific overrides
    mode = config.get('pipeline', {}).get('mode', 'quick')
    if mode in config.get('modes', {}):
        mode_config = config['modes'][mode]
        logger.info(f"Applying mode: {mode}")
        
        # Deep merge mode config into main config
        for key, value in mode_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    logger.info("Configuration loaded successfully")
    return config


def get_config_value(config, *keys, default=None):
    """
    Safely get nested config value.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found
    
    Returns:
        Config value or default
    
    Example:
        get_config_value(config, 'training', 'time_limit', default=600)
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def get_model_config(config):
    """
    Build a unified model configuration from the full config dict.

    - Prefers model['mode'] for behavior:
        'traditional', 'chronos_zero_shot', 'chronos_finetune', ...
    - Still supports legacy model['zero_shot'] == 'yes' if mode not set.
    """
    config = config or {}
    model_config = config.get("model", {})
    forecast_config = config.get("forecast", {}) or {}

    mode = model_config.get("mode")  # new-style mode
    zero_shot_legacy = str(model_config.get("zero_shot", "")).lower()

    # Legacy fallback: if mode not set, derive from zero_shot
    if mode is None:
        if zero_shot_legacy == "yes":
            mode = "chronos_zero_shot"
        else:
            mode = "traditional"

    # Build final config
    return {
        "mode": mode,
        "chronos_preset": model_config.get("chronos_preset", "chronos2_small"),
        "prediction_length": forecast_config["horizon"],
        "freq": forecast_config.get("frequency", "D"),
        "eval_metric": model_config.get("eval_metric", "MASE"),
    }
