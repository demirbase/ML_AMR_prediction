# AMR Prediction with Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![status](https://img.shields.io/badge/status-active-green.svg)]()

A comprehensive pipeline to predict Antimicrobial Resistance (AMR) in bacterial genomes using k-mer frequencies and machine learning.

---

## Project Overview

This repository contains a comprehensive machine learning pipeline for predicting Antimicrobial Resistance (AMR) from bacterial genome data. The pipeline automates the entire workflow from raw data ingestion to model evaluation, designed to be modular, configurable, and reproducible.

The core functionality includes:
- **Data Ingestion**: Processes an initial CSV to identify contigs and determine their beta-lactam resistance status.
- **Genome Acquisition**: Downloads the corresponding genome contigs from the NCBI database.
- **Feature Engineering**: Calculates k-mer frequency profiles for each genome to serve as features.
- **Dataset Creation**: Merges resistance labels with k-mer features and splits the data into stratified training, validation, and test sets.
- **Model Training**: Employs `GridSearchCV` to train and tune multiple classifier models (Random Forest, Gradient Boosting, SVM).
- **Model Evaluation**: Performs a rigorous evaluation of the best model on the test set, generating detailed classification reports and visualizations.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/demirbase/ML_AMR_prediction.git
    cd ML_AMR_prediction
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install external tools:**
    This pipeline relies on `ncbi-acc-download` and `kmertools`. Please ensure they are installed and accessible in your system's `PATH`.

## Configuration

All pipeline parameters are managed in `config/pipeline_config.py`. Before running the pipeline, you must:

1.  **Set your NCBI API Key:** Open `config/pipeline_config.py` and replace `'YOUR_API_KEY_HERE'` with your actual NCBI API key.
    ```python
    'ncbi_api_key': 'YOUR_API_KEY_HERE', # IMPORTANT: Replace this
    ```
2.  **Adjust Paths and Parameters (Optional):** Modify file paths, k-mer size, model hyperparameters, and other settings as needed within this file.

## Usage

The pipeline is controlled via `main.py`. There are three main commands to execute the different stages of the workflow.

### 1. Run the Data Processing Pipeline
This command executes the entire data preparation workflow: it processes the initial resistance info, downloads genomes, calculates k-mer frequencies, merges the datasets, and finally creates the training, validation, and test sets.

```bash
python main.py data
```

### 2. Run the Model Training Pipeline
This command trains all models specified in the configuration file on the training set. It uses GridSearchCV for hyperparameter tuning, evaluates the best model on the validation set, and saves the trained model objects and validation reports.

```bash
python main.py train
```

### 3. Evaluate a Trained Model
This command performs a final, in-depth evaluation of a specified trained model using the test set. It generates a comprehensive report and a plot suite including ROC/PR curves, a confusion matrix, and a calibration curve.

Replace `RandomForest` with the name of the model you wish to evaluate (e.g., `GradientBoosting`, `SVM`).

```bash
python main.py evaluate --model RandomForest
```

## Project Structure

```
ml_amr_prediction/
├── README.md               # This documentation file
├── main.py                 # Main entry point to run the pipelines
├── requirements.txt        # Python package dependencies
├── config/
│   └── pipeline_config.py  # Central configuration for all parameters and paths
└── src/
    ├── data_processing/    # Modules for the data preparation workflow
    │   ├── resistance_info.py    # Processes raw resistance data
    │   ├── genome_downloader.py  # Downloads genomes from NCBI
    │   ├── kmer_processor.py     # Calculates k-mer frequencies
    │   ├── data_merger.py        # Merges k-mers and resistance labels
    │   └── training_set.py       # Creates train/validation/test splits
    └── modeling/               # Modules for the machine learning workflow
        ├── train.py              # Trains and tunes models
        ├── evaluate.py           # Evaluates final models on the test set
        └── predict.py            # Script for making predictions with a trained model