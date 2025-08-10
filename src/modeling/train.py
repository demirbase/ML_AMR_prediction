# src/modeling/train.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class ModelTrainer:
    """
    Handles the training, hyperparameter tuning, and saving of ML models.
    """

    def __init__(self, config: dict):
        self.params = config['model_training_params']
        self.data_dir = Path(self.params['data_dir'])
        self.models_dir = Path(self.params['models_output_dir'])
        self.reports_dir = Path(self.params['reports_output_dir'])
        self.hyperparameters = self.params['hyperparameters']

        # Ensure output directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        """Loads the training and validation datasets."""
        print("Loading training and validation data...")
        train_df = pd.read_csv(self.data_dir / 'train_set.csv')
        val_df = pd.read_csv(self.data_dir / 'validation_set.csv')

        self.X_train = train_df.drop(columns=['Resistance (1/0)'])
        self.y_train = train_df['Resistance (1/0)']
        self.X_val = val_df.drop(columns=['Resistance (1/0)'])
        self.y_val = val_df['Resistance (1/0)']
        print("Data loaded successfully.")

    @staticmethod
    def _get_model(model_name: str):
        """Returns a model instance based on its name. This is a static method."""
        if model_name == 'RandomForest':
            return RandomForestClassifier(random_state=42)
        if model_name == 'GradientBoosting':
            return GradientBoostingClassifier(random_state=42)
        if model_name == 'SVM':
            return SVC(probability=True, random_state=42)
        raise ValueError(f"Model {model_name} not supported.")

    def run_training(self):
        """
        Executes the training pipeline for all models defined in the config.
        """
        self._load_data()

        for model_name in self.params['models_to_run']:
            print(f"\n--- Training {model_name} ---")

            model = self._get_model(model_name)
            param_grid = self.hyperparameters.get(model_name, {})

            print("Performing hyperparameter tuning with GridSearchCV...")
            # Use cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            best_model = grid_search.best_estimator_
            print(f"Best parameters found: {grid_search.best_params_}")

            # Evaluate on validation set
            print("Evaluating best model on validation set...")
            y_pred = best_model.predict(self.X_val)
            report = classification_report(self.y_val, y_pred)
            print(report)

            # Save the report
            report_path = self.reports_dir / f'{model_name}_validation_report.txt'
            with open(report_path, 'w') as f:
                f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
                f.write(report)
            print(f"Validation report saved to {report_path}")

            # Save the trained model
            model_path = self.models_dir / f'best_{model_name.lower()}.joblib'
            joblib.dump(best_model, model_path)
            print(f"Trained model saved to {model_path}")
