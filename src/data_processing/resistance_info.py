# src/data_processing/resistance_info.py

import pandas as pd
from pathlib import Path


class ResistanceInfo:
    """
    Processes a raw data file to extract resistance status for each contig.
    """

    def __init__(self, input_file: Path, output_file: Path):
        """
        Initializes the ResistanceInfo processor.

        Args:
            input_file (Path): Path to the input CSV file (e.g., microbigge3.csv).
            output_file (Path): Path to save the processed CSV file.
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        print(f"ResistanceInfo initialized. Input: {self.input_file}")

    @staticmethod
    def _determine_resistance(subclass_series: pd.Series) -> int:
        """
        Determines resistance based on the presence of 'BETA-LACTAM'.

        Args:
            subclass_series (pd.Series): A series of resistance classes for a contig.

        Returns:
            int: 1 if 'BETA-LACTAM' is present, 0 otherwise.
        """
        return 1 if 'BETA-LACTAM' in subclass_series.values else 0

    def process_data(self):
        """
        Loads data, determines resistance for each contig, and saves the result.
        """
        print("Starting resistance data processing...")
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found at {self.input_file}")

        df = pd.read_csv(self.input_file)

        # Determine resistance status for each contig
        df['Resistance (1/0)'] = df.groupby('Contig')['Class'].transform(self._determine_resistance)

        # Create a clean dataframe with unique contigs and their resistance status
        processed_df = df[['Contig', 'Resistance (1/0)']].drop_duplicates(subset=['Contig'])
        processed_df = processed_df.sort_values(
            by=['Resistance (1/0)', 'Contig'], ascending=[False, True]
        )

        # Ensure the output directory exists and save the file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(self.output_file, index=False)
        print(f"Resistance information saved to {self.output_file}")

