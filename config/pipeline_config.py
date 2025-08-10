# config/pipeline_config.py

"""
Centralized configuration for the entire AMR prediction pipeline.

This file contains all parameters, file paths, and hyperparameters
to ensure that experiments are reproducible and easy to modify.
"""

pipeline_config = {
    # --- General Paths ---
    # NOTE: Paths are relative to the project root directory (ml_amr_predict/).
    'data_dir': 'data',
    'models_dir': 'models',
    'reports_dir': 'reports',

    # --- Data Processing Parameters ---
    'data_processing_params': {
        'raw_data_input': 'data/processed/files_csv/microbigge3.csv',
        'resistance_info_output': 'data/processed/files_csv/betalactam_info_contig.csv',
        'genome_dir': 'data/raw/genomes',
        'kmer_dir': 'data/processed/kmer_densities',
        'final_merged_file': 'data/processed/files_csv/resistance_3-mer.csv',
        'training_set_output_dir': 'data/training_data',
        'ncbi_api_key': 'YOUR_API_KEY_HERE',  # IMPORTANT: Replace with your actual NCBI API key
        'k_size': 3,
        'max_workers': 8,
    },

    # --- Training Set Creation Parameters ---
    'training_set_params': {
        'scale_features': True,
        'test_size': 0.2,
        'val_size': 0.15,  # Proportion of the (1 - test_size) data
        'random_state': 42,
    },

    # --- Model Training Parameters ---
    'model_training_params': {
        'models_to_run': ['RandomForest', 'GradientBoosting', 'SVM'],
        # Hyperparameter grid for GridSearchCV
        'hyperparameters': {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    }
}
