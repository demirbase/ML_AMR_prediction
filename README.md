# AMR Genomic Project

This project predicts antibiotic resistance from genomic data using K-mer counting and XGBoost.

## Project Structure

```text
AMR_Genomic_Project/
├── config/             # Configuration files
├── data/               # Data directory (raw genomes, matrix, kmc outputs)
├── logs/               # Log files
├── models/             # Trained models and Optuna studies
├── results/            # Results (figures, tables)
├── src/                # Source code
│   ├── utils.py        # Utility functions
│   ├── data_processing # Scripts for data preparation
│   ├── modeling        # Scripts for training and tuning
│   └── analysis        # Scripts for feature extraction and visualization
```

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure KMC and KMC Tools are installed and their paths are set in `config/config.yaml`.

## Usage

### 1. Configuration
Edit `config/config.yaml` to set your target antibiotic and adjustments to paths or hyperparameters.

### 2. Run Pipeline

**Step 1: Data Processing (Matrix Generation)**
Generates the sparse matrix from raw genomes using KMC.
```bash
python src/data_processing/02_matrix_generation.py
```

**Step 2: Hyperparameter Tuning (Optional)**
Runs Optuna to find the best XGBoost parameters.
```bash
python src/modeling/03_hyperparam_tuning.py
```

**Step 3: Final Training**
Trains the final model using the best parameters (from config or tuning).
```bash
python src/modeling/04_train_final.py
```

**Step 4: Feature Extraction**
Extracts the top contributing K-mers (features) from the trained model.
```bash
python src/analysis/05_feature_extraction.py
```

**Step 5: Visualization**
Generates performance plots and dashboards.
```bash
python src/analysis/06_visualization.py
```

## Outputs
-   Models are saved in `models/`.
-   Metrics and feature tables are in `results/tables/`.
-   Figures are in `results/figures/`.
-   Logs are in `logs/`.
