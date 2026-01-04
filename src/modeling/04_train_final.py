import pandas as pd
import numpy as np
import xgboost as xgb
import os
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
import sys
import gc
import random
import json

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import load_config, setup_logger

def get_y_chunk(y_all, chunk_id, chunk_size, total_len):
    start = chunk_id * chunk_size
    end = min((chunk_id + 1) * chunk_size, total_len)
    return y_all[start:end]

def final_training_incremental(best_params, train_files, y_all, chunk_size, logger):
    logger.info("\n--- FINAL TRAINING STARTING (INCREMENTAL) ---")
    
    total_trees = best_params.pop('n_estimators')
    trees_per_chunk = max(1, int(np.ceil(total_trees / len(train_files))))
    
    logger.info(f"   -> Total Training Chunks: {len(train_files)}")
    logger.info(f"   -> Target Trees: {total_trees} | Trees Per Chunk: {trees_per_chunk}")
    
    params = best_params.copy()
    model = None
    
    # Ensure params have defaults if not in best_params
    if 'device' not in params: params['device'] = 'cpu'
    if 'n_jobs' not in params: params['n_jobs'] = 8
    
    for i, f in enumerate(train_files):
        logger.info(f"      [{i+1}/{len(train_files)}] Processing: {f.name}")
        
        X_chunk = load_npz(f)
        chunk_num = int(f.stem.split('_')[-1])
        y_chunk = get_y_chunk(y_all, chunk_num, chunk_size, len(y_all))
        
        dtrain = xgb.DMatrix(X_chunk, label=y_chunk, nthread=params['n_jobs'])
        model = xgb.train(params, dtrain, num_boost_round=trees_per_chunk, xgb_model=model)
        
        del X_chunk, y_chunk, dtrain
        gc.collect()
        
    return model

def final_test(model, test_files, y_all, chunk_size, logger, output_metrics_file=None):
    logger.info("\n--- FINAL TEST AND REPORTING ---")
    
    y_true_all = []
    y_prob_all = []
    
    logger.info("   -> Predicting on test data...")
    
    for f in test_files:
        X_chunk = load_npz(f)
        chunk_num = int(f.stem.split('_')[-1])
        y_chunk = get_y_chunk(y_all, chunk_num, chunk_size, len(y_all))
        
        dtest = xgb.DMatrix(X_chunk, nthread=8)
        probs = model.predict(dtest)
        
        y_true_all.extend(y_chunk)
        y_prob_all.extend(probs)
        
        del X_chunk, dtest
        gc.collect()
        
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    # Threshold Optimization
    best_thresh = 0.5
    best_mcc = -1
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (y_prob_all >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true_all, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
            
    logger.info(f"\n   -> Optimized Threshold: {best_thresh:.2f}")
    y_pred_all = (y_prob_all >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f"\n======== REPORT ========")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"ROC AUC  : {auc:.4f}")
    logger.info(f"MCC      : {best_mcc:.4f}")
    logger.info("========================")
    logger.info(f"\nClassification Report:\n{classification_report(y_true_all, y_pred_all)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    if output_metrics_file:
        results = {
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "mcc": float(best_mcc),
            "best_threshold": float(best_thresh),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
            "metrics_text": classification_report(y_true_all, y_pred_all, output_dict=True)
        }
        with open(output_metrics_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved to {output_metrics_file}")

def main():
    config = load_config()
    logger = setup_logger("final_training", log_file=config['paths']['logs_dir'] / "04_train_final.log")
    
    logger.info("--- 1. DATA PREPARATION (RANDOM SHUFFLE) ---")
    
    target_antibiotic = config['project']['target_antibiotic']
    file_prefix = config['training']['file_prefix']
    chunk_size = config['preprocessing']['chunk_size']
    random_seed = config['training']['random_seed']
    test_chunks_count = config['training']['test_chunks']
    
    matrix_dir = Path(config['paths']['matrix_dir'])
    models_dir = Path(config['paths']['models_dir'])
    tables_dir = Path(config['paths']['tables_dir'])
    
    y_path = matrix_dir / f"y_{file_prefix}.csv"
    if not y_path.exists():
        logger.critical("Label file not found!")
        sys.exit(1)
    y_all = pd.read_csv(y_path)['label'].values
    
    all_files = list(matrix_dir.glob(f"X_{file_prefix}_part_*.npz"))
    if not all_files:
        logger.critical("Matrix chunks not found!")
        sys.exit(1)
        
    random.seed(random_seed)
    random.shuffle(all_files)
    
    test_files = all_files[-test_chunks_count:]
    train_files = all_files[:-test_chunks_count]
    
    logger.info(f"   -> Total Chunks: {len(all_files)}")
    logger.info(f"   -> Test Reserved: {[f.stem for f in test_files]}")
    
    # Load Best Params from Config or File?
    # Priority: File > Config
    # If the user ran 03_hyperparam_tuning.py, we might have best_params_{target}.json
    best_params_file = models_dir / f"best_params_{target_antibiotic}.json"
    if best_params_file.exists():
        logger.info(f"Loading best params from {best_params_file}")
        with open(best_params_file, "r") as f:
            best_params = json.load(f)
    else:
        logger.info("Using best params from config.yaml")
        best_params = config['best_params'].copy()
        
    # Final Training
    final_model = final_training_incremental(best_params, train_files, y_all, chunk_size, logger)
    
    # Final Test
    metrics_file = tables_dir / f"metrics_{target_antibiotic}.json"
    tables_dir.mkdir(parents=True, exist_ok=True)
    final_test(final_model, test_files, y_all, chunk_size, logger, output_metrics_file=metrics_file)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"xgboost_{target_antibiotic}_final.json"
    final_model.save_model(model_path)
    logger.info(f"\nModel Saved: {model_path}")

if __name__ == "__main__":
    main()
