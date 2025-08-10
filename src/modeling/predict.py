# src/modeling/predict.py

import pandas as pd
import joblib
from pathlib import Path


class Predictor:
    """
    Loads a saved model to make predictions on new data.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions on a DataFrame of new samples.

        Args:
            input_data (pd.DataFrame): A DataFrame with the same columns as the training data.

        Returns:
            pd.DataFrame: The input data with an added 'Prediction' column.
        """
        predictions = self.model.predict(input_data)
        probabilities = self.model.predict_proba(input_data)[:, 1]  # Probability of class 1 (Resistant)

        result_df = input_data.copy()
        result_df['Prediction'] = predictions
        result_df['Resistance_Probability'] = probabilities
        return result_df


# Example usage:
if __name__ == '__main__':
    # This is a demonstration of how to use the Predictor class.

    # 1. Define the path to your best model
    # (Replace with the actual model you want to use)
    BEST_MODEL_PATH = 'models/best_randomforest.joblib'

    # 2. Create some dummy new data (must have the same columns as training data)
    # In a real scenario, you would generate these k-mer frequencies from a new genome.
    dummy_data = {
        'AAA': [0.04, 0.08], 'AAC': [0.03, 0.03], 'AAG': [0.02, 0.04], 'AAT': [0.03, 0.06],
        'ACA': [0.02, 0.02], 'ACC': [0.03, 0.02], 'ACG': [0.03, 0.01], 'ACT': [0.02, 0.02],
        'AGA': [0.02, 0.02], 'AGC': [0.03, 0.02], 'AGG': [0.02, 0.02], 'ATA': [0.02, 0.02],
        'ATC': [0.03, 0.02], 'ATG': [0.03, 0.03], 'CAA': [0.03, 0.02], 'CAC': [0.02, 0.02],
        'CAG': [0.04, 0.01], 'CCA': [0.03, 0.01], 'CCC': [0.02, 0.03], 'CCG': [0.03, 0.03],
        'CGA': [0.03, 0.05], 'CGC': [0.04, 0.01], 'CTA': [0.01, 0.02], 'CTC': [0.01, 0.03],
        'GAA': [0.03, 0.02], 'GAC': [0.02, 0.02], 'GCA': [0.04, 0.04], 'GCC': [0.03, 0.04],
        'GGA': [0.02, 0.02], 'GTA': [0.02, 0.03], 'TAA': [0.02, 0.02], 'TCA': [0.03, 0.03]
    }
    new_genomes_df = pd.DataFrame(dummy_data)

    try:
        # 3. Create a predictor and make predictions
        predictor = Predictor(model_path=BEST_MODEL_PATH)
        results = predictor.predict(new_genomes_df)

        print("\n--- Prediction Results ---")
        print(results)

    except FileNotFoundError as e:
        print(e)
        print("Please ensure you have trained the models first by running the training pipeline.")

