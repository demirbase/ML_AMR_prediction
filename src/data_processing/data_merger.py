# src/data_processing/data_merger.py

import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class DataMerger:
    """
    Merges k-mer density data with resistance information into a single dataset.
    """

    def __init__(self, resistance_file: Path, kmer_dir: Path, output_file: Path):
        """
        Initializes the DataMerger.

        Args:
            resistance_file (Path): Path to the CSV with contig resistance labels.
            kmer_dir (Path): Directory containing the k-mer count CSV files.
            output_file (Path): Path to save the final merged CSV file.
        """
        self.resistance_file = Path(resistance_file)
        self.kmer_dir = Path(kmer_dir)
        self.output_file = Path(output_file)
        print(f"DataMerger initialized. Output file: {self.output_file}")

    def merge_data(self):
        """
        Merges all k-mer CSVs with the main resistance data file.
        """
        print("Starting data merging process...")
        if not self.resistance_file.exists() or not self.kmer_dir.exists():
            raise FileNotFoundError("Input resistance file or k-mer directory not found.")

        # Load resistance data and set 'Contig' as the index for quick lookups
        resistance_data = pd.read_csv(self.resistance_file)
        resistance_data.set_index('Contig', inplace=True)

        all_kmer_files = [f for f in os.listdir(self.kmer_dir) if f.endswith('.csv')]
        merged_data_list = []

        for filename in tqdm(all_kmer_files, desc="Merging k-mer files"):
            # Extract contig name from filename (e.g., 'CONTIG_3mer.csv' -> 'CONTIG')
            contig_name = filename.split('_')[0]
            if contig_name in resistance_data.index:
                kmer_filepath = self.kmer_dir / filename
                kmer_data = pd.read_csv(kmer_filepath)

                # Convert the single row of k-mer data to a dictionary
                row_data = kmer_data.to_dict(orient='records')[0]
                row_data['Contig'] = contig_name
                row_data['Resistance (1/0)'] = resistance_data.loc[contig_name, 'Resistance (1/0)']
                merged_data_list.append(row_data)

        if not merged_data_list:
            print("Warning: No data was merged. Check contig names and file formats.")
            return

        # Create the final DataFrame
        merged_df = pd.DataFrame(merged_data_list)

        # Reorder columns to have Contig and Resistance first
        cols_to_front = ['Contig', 'Resistance (1/0)']
        other_cols = [c for c in merged_df.columns if c not in cols_to_front]
        final_cols = cols_to_front + sorted(other_cols)  # Sort k-mer columns alphabetically
        merged_df = merged_df[final_cols]

        # Sort the final dataframe
        merged_df = merged_df.sort_values(by=['Resistance (1/0)', 'Contig'], ascending=[False, True])

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(self.output_file, index=False)
        print(f"Merged data successfully saved to {self.output_file}")
