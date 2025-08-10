# src/data_processing/pipeline.py

from pathlib import Path
from .resistance_info import ResistanceInfo
from .genome_downloader import GenomeDownloader
from .kmer_processor import KmerProcessor
from .data_merger import DataMerger
from .training_set import TrainingSetCreator


class AMRPredictionPipeline:
    """
    Orchestrates the entire data processing and training set creation workflow.
    """

    def __init__(self, config: dict):
        """
        Initializes all components of the data processing pipeline.

        Args:
            config (dict): The main configuration dictionary for the project.
        """
        self.config = config
        dpp = config['data_processing_params']

        # Instantiate all processing components with config values
        self.resistance_informer = ResistanceInfo(
            input_file=dpp['raw_data_input'],
            output_file=dpp['resistance_info_output']
        )
        self.genome_downloader = GenomeDownloader(
            csv_file=dpp['resistance_info_output'],
            output_dir=dpp['genome_dir'],
            api_key=dpp['ncbi_api_key'],
            max_workers=dpp['max_workers']
        )
        self.kmer_processor = KmerProcessor(
            input_dir=dpp['genome_dir'],
            output_dir=dpp['kmer_dir'],
            k_size=dpp['k_size'],
            max_workers=dpp['max_workers']
        )
        self.data_merger = DataMerger(
            resistance_file=dpp['resistance_info_output'],
            kmer_dir=dpp['kmer_dir'],
            output_file=dpp['final_merged_file']
        )
        self.training_set_creator = TrainingSetCreator(config=self.config)

        print("\nAMR Data Processing Pipeline Initialized.")

    def run(self):
        """
        Executes the entire data processing pipeline step-by-step.
        """
        print("\n--- [START] Full Data Processing Pipeline Execution ---")

        print("\n--- Step 1: Processing Resistance Info ---")
        self.resistance_informer.process_data()

        print("\n--- Step 2: Downloading Genomes ---")
        self.genome_downloader.download_all_genomes()

        print("\n--- Step 3: Calculating K-mer Densities ---")
        self.kmer_processor.process_all_files()

        print("\n--- Step 4: Merging All Data ---")
        self.data_merger.merge_data()

        print("\n--- Step 5: Creating Training, Validation, and Test Sets ---")
        self.training_set_creator.create_sets()

        print("\n--- [SUCCESS] Full Data Processing Pipeline Finished ---")
