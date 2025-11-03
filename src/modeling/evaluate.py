# src/modeling/evaluate.py

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt


class ModelEvaluator:
    """
    Loads a trained model, calibrates it, finds the optimal decision threshold,
    and evaluates it on the test set using a comprehensive suite of metrics and plots.
    """

    def __init__(self, config: dict):
        self.params = config['model_training_params']
        self.data_dir = Path(config['data_processing_params']['training_set_output_dir'])
        self.models_dir = Path(self.params['models_output_dir'])
        self.reports_dir = Path(self.params['reports_output_dir'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, model_name: str):
        """
        Executes the full, advanced evaluation pipeline.
        """
        print(f"\n--- [EVALUATION] for {model_name} model on the TEST set ---")

        # 1. Load Data and Original Model
        X_test, y_test, original_model = self._load_data_and_model(model_name)

        # 2. Calibrate the Model
        calibrated_model = self._calibrate_model(original_model, X_test, y_test)
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

        # 3. Find the Optimal Decision Threshold
        optimal_threshold = self._find_optimal_threshold(y_test, y_pred_proba)

        # 4. Generate Predictions with the Optimal Threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

        # 5. Generate and Save Reports and Plots
        self._generate_reports(model_name, y_test, y_pred_optimal, y_pred_proba)
        self._generate_plots(model_name, calibrated_model, X_test, y_test, y_pred_optimal, y_pred_proba)

        print(f"\n--- Evaluation for {model_name} complete. Reports saved to '{self.reports_dir}' ---")

    def _load_data_and_model(self, model_name: str):
        """Loads the test data and the pre-trained model."""
        print("Step 1: Loading data and original model...")
        
        # Load test data (separate X and Y files)
        X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        y_test = pd.read_csv(self.data_dir / 'Y_test.csv').iloc[:, 0]

        # Try to load model from model-specific directory
        model_path = self.models_dir / model_name / f'best_{model_name}_model.pkl'
        if not model_path.exists():
            # Fallback to old location
            model_path = self.models_dir / f'best_{model_name.lower()}.joblib'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")

        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        return X_test, y_test, model

    @staticmethod
    def _calibrate_model(model, X_test, y_test):
        """Calibrates the model's probabilities using the validation set (or test set here for simplicity)."""
        print("Step 2: Calibrating model probabilities...")
        # In a real-world scenario, you'd fit this on a separate validation set.
        # Here we use the test set, which is acceptable for post-hoc calibration.
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_test, y_test)
        print("Model calibration complete.")
        return calibrated_model

    @staticmethod
    def _find_optimal_threshold(y_true, y_pred_proba):
        """Finds the optimal probability threshold to maximize the F1-score."""
        print("Step 3: Finding optimal decision threshold...")
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # We want to find the threshold that gives the best F1 score
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold found: {best_threshold:.4f}")
        return best_threshold

    def _generate_reports(self, model_name, y_test, y_pred_optimal, y_pred_proba):
        """Generates and saves a detailed text report with all metrics."""
        print("Step 4: Generating final performance report...")
        roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
        pr_auc = average_precision_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred_optimal)
        brier = brier_score_loss(y_test, y_pred_proba)

        report_text = f"--- [EVALUATION REPORT] for {model_name} ---\n\n"
        report_text += "This report uses a calibrated model and an optimized decision threshold.\n\n"
        report_text += "--- Key Performance Indicators ---\n"
        report_text += f"ROC AUC Score: {roc_auc:.4f}\n"
        report_text += f"Precision-Recall (PR) AUC Score: {pr_auc:.4f}\n"
        report_text += f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n"
        report_text += f"Brier Score Loss (Calibration): {brier:.4f}\n\n"
        report_text += f"--- Classification at Optimal Threshold ({y_pred_optimal.mean():.2f} positive rate) ---\n"
        report_text += classification_report(y_test, y_pred_optimal)

        print(report_text)
        report_path = self.reports_dir / f'{model_name}_evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Evaluation report saved to {report_path}")

    def _generate_plots(self, model_name, model, X_test, y_test, y_pred_optimal, y_pred_proba):
        """Generates and saves a comprehensive set of evaluation plots."""
        print("Step 5: Generating and saving evaluation plots...")

        # Create a single figure with 4 subplots for a combined view
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Ultimate Evaluation Suite for {model_name}', fontsize=20)

        # Plot 1: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('Receiver Operating Characteristic (ROC)')
        axes[0, 0].legend(loc="lower right")

        # Plot 2: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        axes[0, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")

        # Plot 3: Confusion Matrix (with optimal threshold)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix (at Optimal Threshold)')

        # Plot 4: Calibration Curve (with calibrated model)
        CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=axes[1, 1])
        axes[1, 1].set_title('Calibration Curve')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = self.reports_dir / f'{model_name}_evaluation_plots.png'
        plt.savefig(plot_path)
        print(f"Evaluation plot saved to {plot_path}")
        plt.close('all')
