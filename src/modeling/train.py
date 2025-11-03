# src/modeling/train.py

import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve


class ModelTrainer:
    """
    Handles the training and evaluation of ML models using the original project's approach.
    Uses separate X and Y files, QuantileTransformer scaling, CalibratedClassifierCV, 
    and comprehensive evaluation metrics.
    """

    def __init__(self, config: dict):
        self.params = config['model_training_params']
        self.data_dir = Path(config['data_processing_params']['training_set_output_dir'])
        self.models_dir = Path(self.params['models_output_dir'])
        self.reports_dir = Path(self.params['reports_output_dir'])

        # Ensure output directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        """Loads the training, validation, and test datasets using separate X and Y files."""
        print("Loading training, validation, and test data...")
        
        # Load training data
        X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        Y_train = pd.read_csv(self.data_dir / 'Y_train.csv').iloc[:, 0]
        
        # Load validation data
        X_val = pd.read_csv(self.data_dir / 'X_val.csv')
        Y_val = pd.read_csv(self.data_dir / 'Y_val.csv').iloc[:, 0]
        
        # Load test data
        X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        Y_test = pd.read_csv(self.data_dir / 'Y_test.csv').iloc[:, 0]
        
        print("Data loaded successfully:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def _build_model_pipeline(self, model_name: str):
        """
        Builds a complete pipeline with QuantileTransformer and the specified model.
        Returns the pipeline wrapped in CalibratedClassifierCV.
        """
        # Get base model with hyperparameters from config
        if model_name == 'RandomForest':
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=2000,  # Original value: 2000 (use 200 for faster testing)
                max_depth=50,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=False,
                class_weight='balanced_subsample',
                n_jobs=-1  # Use all CPU cores
            )
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5
            )
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif model_name == 'MLPClassifier':
            model = MLPClassifier(
                random_state=42,
                max_iter=1000,
                hidden_layer_sizes=(100, 50)
            )
        elif model_name == 'SVM':
            model = SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # Build pipeline with QuantileTransformer (like original project)
        pipeline = Pipeline([
            ('scaler', QuantileTransformer(output_distribution='normal')),
            ('classifier', model)
        ])
        
        # Wrap with CalibratedClassifierCV (like original project)
        return CalibratedClassifierCV(pipeline, method='sigmoid', cv=10)

    def _evaluate_model(self, model, X, Y):
        """
        Evaluates model and returns comprehensive metrics matching the original project.
        """
        Y_pred = model.predict(X)
        Y_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(Y, Y_pred),
            "precision": precision_score(Y, Y_pred),
            "recall": recall_score(Y, Y_pred),
            "f1_score": f1_score(Y, Y_pred),
            "roc_auc": roc_auc_score(Y, Y_proba),
            "pr_auc": average_precision_score(Y, Y_proba),
            "mcc": matthews_corrcoef(Y, Y_pred),
            "brier_score": brier_score_loss(Y, Y_proba)
        }
        
        confusion_matrix_data = confusion_matrix(Y, Y_pred)
        prob_true, prob_pred = calibration_curve(Y, Y_proba, n_bins=10)
        
        return metrics, confusion_matrix_data, Y, Y_proba, prob_true, prob_pred

    def _save_evaluation_results(self, output_dir, set_name, metrics, confusion_matrix_data, 
                                 Y, Y_proba, prob_true, prob_pred):
        """
        Saves evaluation results to CSV and text files (matching original project format).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix
        pd.DataFrame(confusion_matrix_data).to_csv(
            os.path.join(output_dir, f'{set_name}_confusion_matrix.csv'), index=False
        )
        
        # Save probabilities
        pd.DataFrame({"y_true": Y, "y_proba": Y_proba}).to_csv(
            os.path.join(output_dir, f'{set_name}_proba.csv'), index=False
        )
        
        # Save calibration curve
        calibration_data = pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred})
        calibration_data.to_csv(
            os.path.join(output_dir, f'{set_name}_calibration_curve.csv'), index=False
        )
        
        # Save metrics
        with open(os.path.join(output_dir, f'{set_name}_metrics.txt'), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

    def run_training(self, model_name: str = None):
        """
        Executes the training pipeline for a specific model or all models.
        Follows the original project's approach.
        """
        # Load data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self._load_data()
        
        # Determine which models to train
        models_to_train = [model_name] if model_name else self.params['models_to_run']
        
        for model_name in models_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print('='*60)
            
            # Build and train model
            model = self._build_model_pipeline(model_name)
            print("Fitting model (this may take several minutes)...")
            model.fit(X_train, Y_train)
            print("Model training complete.")
            
            # Create model-specific output directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Evaluate on validation set
            print("Evaluating on validation set...")
            val_results = self._evaluate_model(model, X_val, Y_val)
            self._save_evaluation_results(model_dir, "Validation", *val_results)
            print(f"Validation Accuracy: {val_results[0]['accuracy']:.4f}")
            print(f"Validation F1-Score: {val_results[0]['f1_score']:.4f}")
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_results = self._evaluate_model(model, X_test, Y_test)
            self._save_evaluation_results(model_dir, "Test", *test_results)
            print(f"Test Accuracy: {test_results[0]['accuracy']:.4f}")
            print(f"Test F1-Score: {test_results[0]['f1_score']:.4f}")
            
            # Save the trained model
            model_path = model_dir / f'best_{model_name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            print(f"\n{model_name} training and evaluation complete!")
            print(f"Results saved to {model_dir}\n")
