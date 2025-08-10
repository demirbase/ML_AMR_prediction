# src/data_processing/training_set.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
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

        self.scale_features_flag = tsp.get('scale_features', True)
        self.test_size = tsp.get('test_size', 0.2)
        self.val_size = tsp.get('val_size', 0.15)
        self.random_state = tsp.get('random_state', 42)

        self.scaler = StandardScaler()
        print("TrainingSetCreator initialized.")

    def create_sets(self):
        """
        Executes the full pipeline: load, split, scale, and save data sets.
        """
        print("Loading data for training set creation...")
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file for training set not found at {self.input_file}")

        data = pd.read_csv(self.input_file)

        # Separate features (X) and target (y)
        y = data['Resistance (1/0)']
        x = data.drop(columns=['Contig', 'Resistance (1/0)'])

        print("Splitting data into training, validation, and test sets...")
        # Split into training+validation and test sets
        remaining_size = 1.0 - self.test_size
        val_relative_size = self.val_size / remaining_size

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # Split training+validation into final training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=val_relative_size, stratify=y_train_val, random_state=self.random_state
        )

        if self.scale_features_flag:
            print("Scaling features using StandardScaler...")
            x_train = pd.DataFrame(self.scaler.fit_transform(x_train), columns=x_train.columns)
            x_val = pd.DataFrame(self.scaler.transform(x_val), columns=x_val.columns)
            x_test = pd.DataFrame(self.scaler.transform(x_test), columns=x_test.columns)

        print(f"Saving data sets to {self.output_dir}...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        pd.concat([y_train.reset_index(drop=True), x_train], axis=1).to_csv(
            self.output_dir / "train_set.csv", index=False
        )
        pd.concat([y_val.reset_index(drop=True), x_val], axis=1).to_csv(
            self.output_dir / "validation_set.csv", index=False
        )
        pd.concat([y_test.reset_index(drop=True), x_test], axis=1).to_csv(
            self.output_dir / "test_set.csv", index=False
        )

        print("Training, validation, and test sets created successfully.")
