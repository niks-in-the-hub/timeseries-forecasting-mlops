"""
Preprocessing functions for retail sales forecasting.
Handles data cleaning, merging, and formatting for AutoGluon Chronos.
"""

import pandas as pd
import numpy as np
import logging
from src.utils import load_train_data, load_store_data, validate_columns, get_date_range


# ============================================================================
# DATA LOADING AND MERGING
# ============================================================================

def load_and_merge_data():
    """
    Load train and store data, then merge them together.
    
    Returns:
        Merged DataFrame with sales and store information
    """
    logger = logging.getLogger(__name__)
    
    # Load both datasets
    train_df = load_train_data()
    store_df = load_store_data()
    
    # Merge on Store column
    logger.info("Merging train and store data...")
    merged_df = train_df.merge(store_df, on='Store', how='left')
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    return merged_df


# DATA CLEANING

def clean_data(df):
    """
    Clean the merged dataset:
    - Convert dates to datetime
    - Filter out closed stores
    - Handle missing values
    - Sort by store and date
    
    Args:
        df: Merged DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter out days when store was closed (no sales)
    # We only want to forecast for open stores
    initial_rows = len(df)
    df = df[df['Open'] == 1].copy()
    logger.info(f"Filtered out {initial_rows - len(df)} closed store days")
    
    # Remove rows where Sales is missing or zero
    # (can't learn from zero sales days)
    df = df[df['Sales'] > 0].copy()
    
    # Sort by Store and Date (important for time series!)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    logger.info(f"Cleaned data shape: {df.shape}")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        DataFrame with missing values handled
    """
    logger = logging.getLogger(__name__)
    
    df = df.copy()
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
    
    # Fill missing CompetitionDistance with a large value (no nearby competition)
    if 'CompetitionDistance' in df.columns:
        df['CompetitionDistance'].fillna(999999, inplace=True)
    
    # Fill other missing values with 0
    df.fillna(0, inplace=True)
    
    return df


# FORMAT FOR AUTOGLUON CHRONOS

def format_for_chronos(df):
    """
    Format data for AutoGluon Chronos.
    Required columns: timestamp, target, item_id
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        DataFrame in Chronos format with additional static features
    """
    logger = logging.getLogger(__name__)
    logger.info("Formatting data for AutoGluon Chronos...")
    
    # Create the required columns for Chronos
    chronos_df = pd.DataFrame({
        'timestamp': df['Date'],          # Required: datetime column
        'target': df['Sales'],            # Required: what we're forecasting
        'item_id': df['Store'].astype(str)  # Required: identifier for each time series
    })
    
    # Add static features from store data (these don't change over time)
    # AutoGluon can use these to improve forecasts
    static_features = [
        'StoreType',
        'Assortment',
        'CompetitionDistance',
        'Promo'  # This one varies but we'll include it
    ]
    
    for feature in static_features:
        if feature in df.columns:
            chronos_df[feature] = df[feature]
    
    logger.info(f"Chronos format shape: {chronos_df.shape}")
    logger.info(f"Columns: {list(chronos_df.columns)}")
    logger.info(f"Number of unique stores: {chronos_df['item_id'].nunique()}")
    
    return chronos_df


# TRAIN/VALIDATION SPLIT

def create_train_val_split(df, val_days=None):
    """
    Split data into train and validation sets for time series.
    Uses last N days for validation (realistic for forecasting).
    
    Args:
        df: DataFrame in Chronos format
        val_days: Number of days to use for validation (from config)
    
    Returns:
        Tuple of (train_df, val_df)
    """
    if val_days is None:
        raise ValueError("val_days must be provided from config")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating train/validation split (last {val_days} days for validation)...")
    
    df = df.copy()
    
    # Get the date range
    min_date, max_date = get_date_range(df, 'timestamp')
    logger.info(f"Date range: {min_date} to {max_date}")
    
    # Calculate split date (validation starts here)
    split_date = max_date - pd.Timedelta(days=val_days)
    logger.info(f"Split date: {split_date}")
    
    # Split the data
    train_df = df[df['timestamp'] < split_date].copy()
    val_df = df[df['timestamp'] >= split_date].copy()
    
    logger.info(f"Train shape: {train_df.shape} ({min_date} to {split_date})")
    logger.info(f"Validation shape: {val_df.shape} ({split_date} to {max_date})")
    
    # Validate that we have data in both splits
    if len(train_df) == 0:
        raise ValueError("Training set is empty!")
    if len(val_df) == 0:
        raise ValueError("Validation set is empty!")
    
    return train_df, val_df


# ============================================================================
# STORE SELECTION (FOR FASTER TESTING)
# ============================================================================

def select_stores(df, num_stores=None, store_ids=None):
    """
    Select a subset of stores for faster experimentation.
    Useful for testing before running on all 1000+ stores.
    
    Args:
        df: DataFrame with item_id column
        num_stores: Number of stores to randomly select (optional)
        store_ids: Specific store IDs to select (optional)
    
    Returns:
        DataFrame with selected stores only
    """
    logger = logging.getLogger(__name__)
    
    if store_ids is not None:
        # Select specific stores
        store_ids = [str(s) for s in store_ids]  # Convert to strings
        df = df[df['item_id'].isin(store_ids)].copy()
        logger.info(f"Selected {len(store_ids)} specific stores")
    
    elif num_stores is not None:
        # Randomly select N stores
        available_stores = df['item_id'].unique()
        if num_stores > len(available_stores):
            logger.warning(f"Requested {num_stores} stores but only {len(available_stores)} available")
            num_stores = len(available_stores)
        
        selected_stores = np.random.choice(available_stores, size=num_stores, replace=False)
        df = df[df['item_id'].isin(selected_stores)].copy()
        logger.info(f"Randomly selected {num_stores} stores")
    
    else:
        # Use all stores
        logger.info(f"Using all {df['item_id'].nunique()} stores")
    
    return df


# FULL PREPROCESSING PIPELINE

def preprocess_pipeline(config=None, num_stores=None, val_days=None):
    """
    Run the full preprocessing pipeline.
    
    Args:
        config: Configuration dictionary (optional)
        num_stores: Number of stores to use (overrides config)
        val_days: Number of days for validation (overrides config)
    
    Returns:
        Tuple of (train_df, val_df) ready for AutoGluon
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Starting preprocessing pipeline")
    logger.info("="*60)
    
    # Load config if not provided
    if config is None:
        from utils import load_config
        config = load_config()
    
    # Get settings from config (allow overrides)
    if num_stores is None:
        num_stores = config.get('training', {}).get('num_stores')
    if val_days is None:
        val_days = config.get('forecast', {}).get('validation_days')
    
    # Step 1: Load and merge
    df = load_and_merge_data()
    
    # Step 2: Clean
    df = clean_data(df)
    df = handle_missing_values(df)
    
    # Step 3: Format for Chronos
    df = format_for_chronos(df)
    
    # Step 4: Select stores (if specified)
    if num_stores is not None:
        df = select_stores(df, num_stores=num_stores)
    
    # Step 5: Train/val split
    train_df, val_df = create_train_val_split(df, val_days=val_days)
    
    logger.info("="*60)
    logger.info("Preprocessing complete!")
    logger.info(f"Final train shape: {train_df.shape}")
    logger.info(f"Final validation shape: {val_df.shape}")
    logger.info("="*60)
    
    return train_df, val_df


# HELPER: PREVIEW DATA

def preview_data(df, n_stores=3):
    """
    Preview data for a few stores to verify formatting.
    Useful for debugging.
    
    Args:
        df: DataFrame to preview
        n_stores: Number of stores to show
    """
    logger = logging.getLogger(__name__)
    
    unique_stores = df['item_id'].unique()[:n_stores]
    
    logger.info("\n" + "="*60)
    logger.info(f"Data Preview (first {n_stores} stores)")
    logger.info("="*60)
    
    for store in unique_stores:
        store_data = df[df['item_id'] == store].head()
        logger.info(f"\nStore {store}:")
        logger.info(f"\n{store_data.to_string()}\n")