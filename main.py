# main.py

import argparse
import sys
from pathlib import Path

# Ensure the source directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config.pipeline_config import pipeline_config
from src.data_processing.pipeline import AMRPredictionPipeline
from src.modeling.train import ModelTrainer
from src.modeling.evaluate import ModelEvaluator


def main():
    """
    Main entry point for the AMR prediction project.

    Handles command-line arguments to run either the data processing
    pipeline or the model training/evaluation pipelines.
    """
    parser = argparse.ArgumentParser(
        description="AMR Prediction Pipeline Orchestrator"
    )
    parser.add_argument(
        'pipeline',
        choices=['data', 'train', 'evaluate'],
        help=(
            "Which pipeline to run: 'data' for processing, "
            "'train' for model training, or 'evaluate' for testing a model."
        )
    )
    parser.add_argument(
        '--model',
        type=str,
        help=(
            "The model to train or evaluate (e.g., 'RandomForest'). "
            "Optional for 'train' (trains all if not specified), required for 'evaluate'."
        )
    )

    args = parser.parse_args()

    try:
        if args.pipeline == 'data':
            print("=============================================")
            print("= Running Data Processing Pipeline          =")
            print("=============================================")
            pipeline = AMRPredictionPipeline(config=pipeline_config)
            pipeline.run()

        elif args.pipeline == 'train':
            print("=============================================")
            print("= Running Model Training Pipeline           =")
            print("=============================================")
            trainer = ModelTrainer(config=pipeline_config)
            trainer.run_training(model_name=args.model)

        elif args.pipeline == 'evaluate':
            if not args.model:
                print("Error: --model argument is required for the 'evaluate' pipeline.")
                sys.exit(1)
            print("=============================================")
            print(f"= Running Final Evaluation for {args.model} =")
            print("=============================================")
            evaluator = ModelEvaluator(config=pipeline_config)
            evaluator.run_evaluation(model_name=args.model)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        # For debugging, you might want to re-raise the exception
        # raise e


if __name__ == '__main__':
    main()
