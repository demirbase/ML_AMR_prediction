import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from pathlib import Path
from scipy.sparse import load_npz, vstack
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
import joblib
import sys
import gc
import random

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import load_config, setup_logger

def get_y_chunk(y_all, chunk_id, chunk_size, total_len):
    start = chunk_id * chunk_size
    end = min((chunk_id + 1) * chunk_size, total_len)
    return y_all[start:end]

def analyze_and_select_chunks(y_all, train_pool_files, chunk_size, logger):
    logger.info("\n--- 1. SMART CHUNK SELECTION (SAFE MODE) ---")
    
    chunk_stats = []
    
    for f in train_pool_files:
        chunk_num = int(f.stem.split('_')[-1])
        y_chunk = get_y_chunk(y_all, chunk_num, chunk_size, len(y_all))
        
        pos_count = sum(y_chunk)
        total_count = len(y_chunk)
        ratio = pos_count / total_count if total_count > 0 else 0
        
        chunk_stats.append({
            'file': f,
            'id': chunk_num,
            'pos_count': pos_count,
            'total': total_count,
            'ratio': ratio
        })
    
    df_stats = pd.DataFrame(chunk_stats).sort_values(by='pos_count', ascending=False)
    
    logger.info("   -> Chunk Analysis (Leaderboard):")
    logger.info(f"\n{df_stats[['id', 'pos_count', 'ratio']].head(3).to_string(index=False)}")
    
    # Strategy: 1 Top + 1 Random
    top_1_file = df_stats.head(1)['file'].tolist()
    
    remaining = df_stats.iloc[1:]
    if len(remaining) >= 1:
        random_file = remaining.sample(n=1, random_state=42)['file'].tolist()
    else:
        random_file = []
        
    selected_files = top_1_file + random_file
    
    logger.info(f"\n   -> SELECTED CHUNKS: {[f.stem for f in selected_files]}")
    logger.info("   -> Reason: 1 Leader + 1 Random (Max RAM Safety).")
    
    return selected_files

def load_data_for_optuna(selected_files, y_all, chunk_size, logger):
    logger.info("   -> Loading Optuna data into RAM...")
    X_list = []
    y_list = []
    
    for f in selected_files:
        X_chunk = load_npz(f)
        chunk_num = int(f.stem.split('_')[-1])
        y_chunk = get_y_chunk(y_all, chunk_num, chunk_size, len(y_all))
        
        X_list.append(X_chunk)
        y_list.append(y_chunk)
        
    X_opt = vstack(X_list)
    y_opt = np.concatenate(y_list)
    
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X_opt, y_opt, test_size=0.2, random_state=42, stratify=y_opt)
    
    logger.info(f"   -> Optuna Train: {X_t.shape} | Val: {X_v.shape}")
    
    dtrain = xgb.DMatrix(X_t, label=y_t, nthread=8)
    dval = xgb.DMatrix(X_v, label=y_v, nthread=8)
    
    return dtrain, dval

def objective(trial, dtrain, dval, base_params):
    params = base_params.copy()
    
    # Suggest hyperparameters
    params.update({
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
    })
    
    n_rounds = trial.suggest_int('n_estimators', 100, 600)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=n_rounds,
        evals=[(dval, "Validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return model.best_score

def main():
    config = load_config()
    logger = setup_logger("hyperparam_tuning", log_file=config['paths']['logs_dir'] / "03_hyperparam_tuning.log")
    
    logger.info("--- HYPERPARAMETER TUNING (OPTUNA) ---")
    
    # Config Params
    target_antibiotic = config['project']['target_antibiotic']
    file_prefix = config['training']['file_prefix']
    chunk_size = config['preprocessing']['chunk_size']
    n_trials = config['training']['n_trials']
    test_chunks = config['training']['test_chunks']
    
    matrix_dir = Path(config['paths']['matrix_dir'])
    models_dir = Path(config['paths']['models_dir'])
    
    # 1. Load Data Info
    y_path = matrix_dir / f"y_{file_prefix}.csv"
    if not y_path.exists():
        logger.critical("Label file not found!")
        sys.exit(1)
        
    y_all = pd.read_csv(y_path)['label'].values
    
    all_files = sorted(list(matrix_dir.glob(f"X_{file_prefix}_part_*.npz")), key=lambda x: int(x.stem.split('_')[-1]))
    if not all_files:
        logger.critical("Matrix chunks not found!")
        sys.exit(1)
        
    # Split
    if len(all_files) <= test_chunks:
        logger.warning("Not enough chunks for separate test set, using last chunk as test")
        test_chunks = 1
        
    train_pool_files = all_files[:-test_chunks]
    
    logger.info(f"   -> Total Chunks: {len(all_files)}")
    logger.info(f"   -> Train Pool: {len(train_pool_files)} | Test Reserved: {test_chunks}")
    
    # 2. Select Subset for Optuna
    selected_files = analyze_and_select_chunks(y_all, train_pool_files, chunk_size, logger)
    dtrain_opt, dval_opt = load_data_for_optuna(selected_files, y_all, chunk_size, logger)
    
    # 3. Optimize
    logger.info(f"\n--- 2. OPTUNA OPTIMIZATION ({n_trials} TRIALS) ---")
    study = optuna.create_study(direction='maximize')
    
    base_params = config['xgboost_params']
    
    study.optimize(lambda trial: objective(trial, dtrain_opt, dval_opt, base_params), n_trials=n_trials)
    
    logger.info("\n   -> Best Parameters Found:")
    logger.info(study.best_params)
    
    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, models_dir / f"optuna_study_{target_antibiotic}_smart.pkl")
    
    # Save best params to a json for the next step
    import json
    best_params_path = models_dir / f"best_params_{target_antibiotic}.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    logger.info(f"Optuna study saved to {models_dir}")
    logger.info(f"Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
