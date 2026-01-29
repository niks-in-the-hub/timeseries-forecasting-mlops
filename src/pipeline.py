"""
Luigi orchestration for retail forecasting pipeline.
Simple task-based workflow orchestration with config support.
"""

import luigi
import logging
import pickle
from pathlib import Path
from utils import setup_logging, create_output_dir, load_config, get_current_timestamp
from preprocess import preprocess_pipeline
from train import train_pipeline
from predict import predict_pipeline


# LUIGI TASKS

class PreprocessTask(luigi.Task):
    """
    Task 1: Preprocess the data.
    """
    run_id = luigi.Parameter()
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return {
            'train': luigi.LocalTarget(str(output_dir / f'train_data_{self.run_id}.pkl')),
            'val': luigi.LocalTarget(str(output_dir / f'val_data_{self.run_id}.pkl'))
        }
    
    def run(self):
        """Run preprocessing."""
        logger = setup_logging()
        config = load_config()
        
        logger.info("="*70)
        logger.info(f"LUIGI TASK: Preprocessing (run_id={self.run_id})")
        logger.info("="*70)
        
        # Run preprocessing with config
        train_df, val_df = preprocess_pipeline(config=config)
        
        # Save outputs (BINARY mode for pickle)
        with open(self.output()['train'].path, 'wb') as f:
            pickle.dump(train_df, f)
        
        with open(self.output()['val'].path, 'wb') as f:
            pickle.dump(val_df, f)
        
        logger.info(" Preprocessing complete")


class TrainTask(luigi.Task):
    """
    Task 2: Train the model. Only required when zero_shot is disabled.
    Depends on PreprocessTask.
    """
    run_id = luigi.Parameter()  # Unique run identifier
    
    def requires(self):
        """This task requires PreprocessTask to complete first."""
        return PreprocessTask(run_id=self.run_id)
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return {
            'model_path': luigi.LocalTarget(str(output_dir / f'model_path_{self.run_id}.txt')),
            'metrics': luigi.LocalTarget(str(output_dir / f'metrics_{self.run_id}.pkl'))
        }
    
    def run(self):
        """Run training."""
        logger = setup_logging()
        config = load_config()
        
        logger.info("="*70)
        logger.info(f"LUIGI TASK: Training (run_id={self.run_id})")
        logger.info("="*70)
        
        # Load preprocessed data (BINARY mode for pickle)
        with open(self.input()['train'].path, 'rb') as f:
            train_df = pickle.load(f)
        
        with open(self.input()['val'].path, 'rb') as f:
            val_df = pickle.load(f)
        
        predictor, metrics, model_path = train_pipeline(
            train_df=train_df,
            val_df=val_df,
            config=config
        )
        
        # Save outputs
        with self.output()['model_path'].open('w') as f:
            f.write(model_path)
        
        with open(self.output()['metrics'].path, 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info("   Training complete")


class PredictTask(luigi.Task):
    """
    Task 3: Generate predictions.
    Depends on TrainTask OR PreprocessTask (if zero-shot).
    """
    run_id = luigi.Parameter()  # Unique run identifier
    
    def requires(self):
        """This task requires TrainTask OR PreprocessTask (if zero-shot)."""
        config = load_config()
        zero_shot_raw = config["model"].get("zero_shot", False)
        zero_shot = str(zero_shot_raw).lower() in ("yes", "true", "1")

        if zero_shot:
            # Zero-shot: skip training, only need preprocessing
            return PreprocessTask(run_id=self.run_id)
        else:
            # Training mode: need trained model
            return TrainTask(run_id=self.run_id)
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return luigi.LocalTarget(str(output_dir / f'predictions_path_{self.run_id}.txt'))
    
    def run(self):
        """Run prediction."""
        logger = setup_logging()
        config = load_config()

        logger.info("=" * 70)
        logger.info(f"LUIGI TASK: Prediction (run_id={self.run_id})")
        logger.info("=" * 70)

        # Normalize zero_shot from config ("yes"/"no" -> bool)
        zero_shot_raw = config["model"].get("zero_shot", "no")
        zero_shot = str(zero_shot_raw).lower() in ("yes", "true", "1")
        logger.info(f"zero_shot_raw={zero_shot_raw}, zero_shot_bool={zero_shot}")

        # Load validation data (always needed)
        preprocess_task = PreprocessTask(run_id=self.run_id)
        with open(preprocess_task.output()["val"].path, "rb") as f:
            val_df = pickle.load(f)

        if zero_shot:
            # Zero-shot path: use a symbolic model id (no trained model on disk)
            zero_shot_model = config["model"].get(
                "zero_shot_model",
                config["model"].get("chronos_preset", "chronos2_small"),
            )
            model_path = f"zero_shot_{zero_shot_model}"
            logger.info(f"Zero-shot prediction using model identifier: {model_path}")
        else:
            # Training path: load model path produced by TrainTask
            with self.input()["model_path"].open("r") as f:
                model_path = f.read().strip()
            logger.info(f"Using trained model from path: {model_path}")

        # Generate predictions (pass zero_shot flag through)
        predictions_df, summary, save_path = predict_pipeline(
            model_path=model_path,
            data=val_df,
            save_output=True,
            zero_shot=zero_shot,
            config=config,
        )

        # Save output path (CSV location) for downstream tasks
        with self.output().open("w") as f:
            f.write(save_path)

        mode_str = "ZERO-SHOT" if zero_shot else "TRAINED"
        logger.info("Predictions complete")
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETE ({mode_str} MODE)!")
        logger.info(f"Predictions saved to: {save_path}")
        logger.info("=" * 70)



# WRAPPER TASK (MAIN PIPELINE)

class ForecastingPipeline(luigi.Task):
    """
    Main pipeline task that runs all steps based on config.
    """
    run_id = luigi.Parameter()  # Unique run identifier
    
    def requires(self):
        """Require the final task (which will trigger all dependencies)."""
        return PredictTask(run_id=self.run_id)
    
    def output(self):
        """Pipeline completion marker."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return luigi.LocalTarget(str(output_dir / f'pipeline_complete_{self.run_id}.txt'))
    
    def run(self):
       """Mark pipeline as complete."""
       logger = setup_logging()
       config = load_config()
       
       # Load results
       with self.input().open('r') as f:
           predictions_path = f.read().strip()
       
       # Normalize zero_shot for mode string
       zero_shot_raw = config['model'].get('zero_shot', False)
       zero_shot = str(zero_shot_raw).lower() in ("yes", "true", "1")
       mode_str = "ZERO-SHOT" if zero_shot else "TRAINED"
       
       # Write completion marker
       with self.output().open('w') as f:
           f.write(f"Pipeline completed successfully! ({mode_str} mode)\n")
           f.write(f"Run ID: {self.run_id}\n")
           f.write(f"Predictions: {predictions_path}\n")
       
       logger.info("\n" + "="*70)
       logger.info(f"ENTIRE PIPELINE COMPLETED SUCCESSFULLY ({mode_str})")
       logger.info(f"Run ID: {self.run_id}")
       logger.info(f"Predictions: {predictions_path}")
       logger.info("="*70 + "\n")