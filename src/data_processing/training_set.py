# src/data_processing/training_set.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


class TrainingSetCreator:
    """
    Creates and preprocesses training, validation, and test sets from the merged data.
    """

    def __init__(self, config: dict):
        """
        Initializes the TrainingSetCreator.

        Args:
            config (dict): The main pipeline configuration dictionary.
        """
        dpp = config['data_processing_params']
        tsp = config['training_set_params']

        self.input_file = Path(dpp['final_merged_file'])
        self.output_dir = Path(dpp['training_set_output_dir'])

        self.random_state = tsp.get('random_state', 42)

        print("TrainingSetCreator initialized.")

    def create_sets(self):
        """
        Executes the full pipeline: load, split, and save data sets.
        Uses the original project's data format (separate X and Y files).
        """
        print("Loading data for training set creation...")
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file for training set not found at {self.input_file}")

        data = pd.read_csv(self.input_file)

        # Separate features (X) and target (Y)
        Y = data['Resistance (1/0)']
        X = data.iloc[:, 2:]  # 3-mer densities (all columns after Contig and Resistance)

        print("Splitting data into training, validation, and test sets...")
        # First split: 70% train, 30% temp (for validation + test)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.3, stratify=Y, random_state=self.random_state
        )

        # Second split: Split temp into 50% validation, 50% test (15% each of total)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=self.random_state
        )

        print(f"Saving data sets to {self.output_dir}...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets in separate X and Y files (original format)
        X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        X_val.to_csv(self.output_dir / "X_val.csv", index=False)
        Y_train.to_csv(self.output_dir / "Y_train.csv", index=False)
        Y_test.to_csv(self.output_dir / "Y_test.csv", index=False)
        Y_val.to_csv(self.output_dir / "Y_val.csv", index=False)

        print("Training sets created successfully:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
