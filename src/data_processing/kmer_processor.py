# src/data_processing/kmer_processor.py

import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class KmerProcessor:
    """
    Calculates k-mer densities for genome files using an external tool.
    """

    def __init__(self, input_dir: Path, output_dir: Path, k_size: int, max_workers: int):
        """
        Initializes the KmerProcessor.

        Args:
            input_dir (Path): Directory containing input FASTA files.
            output_dir (Path): Directory to save the output k-mer CSV files.
            k_size (int): The size of k-mers to calculate.
            max_workers (int): The maximum number of concurrent processes.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.k_size = k_size
        self.max_workers = max_workers or os.cpu_count()
        print(f"KmerProcessor initialized. k-size: {self.k_size}")

    def _process_single_file(self, filename: str):
        """
        Processes a single FASTA file to calculate k-mer counts using kmertools.

        Args:
            filename (str): The name of the FASTA file to process.
        """
        input_path = self.input_dir / filename
        output_path = self.output_dir / f"{input_path.stem}_{self.k_size}mer.csv"

        if output_path.exists():
            return  # Skip if already processed

        command = [
            "kmertools", "comp", "oligo",
            "--input", str(input_path),
            "--output", str(output_path),
            "--k-size", str(self.k_size),
            "--preset", "csv", "--header"
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {filename}. Error: {e.stderr}")

    def process_all_files(self):
        """
        Processes all FASTA files in the input directory in parallel.
        """
        print(f"Starting k-mer density calculation for k={self.k_size}...")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Genome input directory not found at {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fasta_files = [f for f in os.listdir(self.input_dir) if f.endswith((".fasta", ".fna"))]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(
                executor.map(self._process_single_file, fasta_files),
                total=len(fasta_files),
                desc=f"Calculating {self.k_size}-mers"
            ))
        print("K-mer processing complete.")
