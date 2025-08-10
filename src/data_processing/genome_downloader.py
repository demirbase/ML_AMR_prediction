# src/data_processing/genome_downloader.py

import subprocess
import time
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class GenomeDownloader:
    """
    Downloads genome files from NCBI based on a list of contig IDs.
    """

    def __init__(self, csv_file: Path, output_dir: Path, api_key: str, max_workers: int):
        """
        Initializes the GenomeDownloader.

        Args:
            csv_file (Path): Path to the CSV file containing a 'Contig' column.
            output_dir (Path): Directory to save the downloaded FASTA files.
            api_key (str): NCBI API key for authentication.
            max_workers (int): The maximum number of concurrent download threads.
        """
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.max_workers = max_workers
        print(f"GenomeDownloader initialized. Output directory: {self.output_dir}")

    def _download_single_genome(self, contig: str):
        """
        Downloads a single genome using ncbi-acc-download with exponential backoff.

        Args:
            contig (str): The contig accession ID to download.
        """
        output_path = self.output_dir / f"{contig}.fasta"
        if output_path.exists():
            # Skip download if file already exists
            return

        retries = 5
        command = [
            "ncbi-acc-download", "--api-key", self.api_key,
            "--format", "fasta", contig, "--out", str(output_path)
        ]

        for attempt in range(retries):
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return  # Success
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed for {contig}. Error: {e.stderr.strip()}")
                if attempt < retries - 1:
                    # Exponential backoff with jitter
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                else:
                    print(f"Failed to download {contig} after {retries} attempts.")

    def download_all_genomes(self):
        """
        Downloads all genomes from the input CSV file in parallel.
        """
        print("Starting genome downloads...")
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Contig CSV file not found at {self.csv_file}")

        data = pd.read_csv(self.csv_file)
        contigs = data['Contig'].unique().tolist()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(
                executor.map(self._download_single_genome, contigs),
                total=len(contigs),
                desc="Downloading genomes"
            ))
        print("Genome download process complete.")
