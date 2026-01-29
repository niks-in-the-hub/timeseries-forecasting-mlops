"""
Main entry point for the retail forecasting pipeline with Luigi orchestration.
Reads configuration from config.yaml.
"""

import sys
import luigi
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import ForecastingPipeline
from utils import load_config, setup_logging, get_current_timestamp


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE - LUIGI ORCHESTRATION")
    print("="*70)
    print("\nUsage:")
    print("  python run.py [mode]")
    print("\nAvailable modes (edit in config.yaml):")
    print("  quick      - Quick test (3 stores, 2 mins)")
    print("  medium     - Medium run (10 stores, 10 mins)")
    print("  production - Production (all stores, 30 mins)")
    print("\nConfiguration:")
    print("  Edit config.yaml to change settings")
    print("  Change 'pipeline: mode:' to switch modes")
    print("  Set 'zero_shot: yes' for instant forecasts")
    print("\nNote:")
    print("  Each run creates new outputs with timestamps")
    print("  Old runs are preserved in luigi_outputs/")
    print("="*70 + "\n")


def main():
    """Run the pipeline using Luigi with config.yaml."""
    
    logger = setup_logging()
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print("\n    Error: Could not load config.yaml")
        print("Make sure config.yaml exists in the project root.")
        sys.exit(1)
    
    # Get mode from command line or config
    if len(sys.argv) > 1:
        requested_mode = sys.argv[1].lower()
        
        # Validate mode
        available_modes = list(config.get('modes', {}).keys())
        if requested_mode not in available_modes:
            print(f"\n    Error: Unknown mode '{requested_mode}'")
            print(f"Available modes: {', '.join(available_modes)}")
            print("\nEdit config.yaml and change 'pipeline: mode:' to one of the available modes.")
            print_usage()
            sys.exit(1)
        
        # Update config with requested mode
        config['pipeline']['mode'] = requested_mode
        
        # Re-apply mode settings
        if requested_mode in config.get('modes', {}):
            mode_config = config['modes'][requested_mode]
            for key, value in mode_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        
        # Save updated mode back (temporarily for this run)
        logger.info(f"Using mode: {requested_mode}")
    else:
        requested_mode = config['pipeline']['mode']
        logger.info(f"Using default mode from config: {requested_mode}")
    
    # Generate unique run ID (timestamp-based)
    run_id = get_current_timestamp()
    logger.info(f"Generated run ID: {run_id}")
    
    # Display configuration
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE WITH LUIGI")
    print("="*70)
    print(f"Run ID: {run_id}")
    print(f"Mode: {requested_mode.upper()}")
    print(f"Zero-shot: {'YES' if config['model']['zero_shot'] else 'NO'}")
    if config['model']['zero_shot']:
        print(f"Model: {config['model']['zero_shot_model']}")
    else:
        print(f"Training time: {config['training']['time_limit']}s")
        print(f"Presets: {config['training']['presets']}")
        print(f"Stores: {config['training']['num_stores'] or 'ALL'}")
    print(f"Forecast horizon: {config['forecast']['horizon']} days")
    print("="*70 + "\n")
    
    # Run the Luigi pipeline with unique run_id
    try:
        success = luigi.build(
            [ForecastingPipeline(run_id=run_id)],
            local_scheduler=True
        )
        
        if success:
            print("\n" + "="*70)
            print("   PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nRun ID: {run_id}")
            print("\nOutputs:")
            print(f"  - Predictions: {config['output']['predictions_dir']}/")
            print(f"  - Models: {config['output']['models_dir']}/")
            print(f"  - Luigi state: {config['output']['luigi_dir']}/")
            print("\nNext steps:")
            print("  - View predictions: ls -la outputs/")
            print("  - View MLflow: mlflow ui")
            print("  - Change settings: edit config.yaml")
            print("  - Run again: python run.py (creates new run)")
            print("="*70 + "\n")
            return 0
        else:
            print("\n" + "="*70)
            print("    PIPELINE FAILED")
            print("="*70)
            print("\nCheck the logs above for details.")
            print("="*70 + "\n")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)