import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import load_config, setup_logger

def main():
    config = load_config()
    logger = setup_logger("feature_extraction", log_file=config['paths']['logs_dir'] / "05_feature_extraction.log")
    
    target_antibiotic = config['project']['target_antibiotic']
    logger.info(f"--- FEATURE EXTRACTION: {target_antibiotic.upper()} ---")
    
    models_dir = Path(config['paths']['models_dir'])
    matrix_dir = Path(config['paths']['matrix_dir'])
    output_dir = Path(config['paths']['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    top_n = config['analysis']['top_n_features']
    model_path = models_dir / f"xgboost_{target_antibiotic}_final.json"
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. Trying fallback name...")
        # Fallback for compatibility if user didn't run 04 yet but has old model?
        # But we are in fresh structure. 
        # Check if user moved 'xgboost_ciprofloxacin_smart_final.json' to models dir manually?
        # The user command copied contents of `models/` to `AMR_Genomic_Project/models/`.
        # Old name was `xgboost_{TARGET_ANTIBIOTIC}_smart_final.json` (from 03_train_model) or `xgboost_{TARGET}_final_v2.json` (from 03_train_final_direct)
        possible_names = [
            f"xgboost_{target_antibiotic}_final.json",
            f"xgboost_{target_antibiotic}_smart_final.json",
            f"xgboost_{target_antibiotic}_final_v2.json"
        ]
        found = False
        for name in possible_names:
            if (models_dir / name).exists():
                model_path = models_dir / name
                found = True
                break
        
        if not found:
            # Fallback check for _final_v2.json
            v2_path = models_dir / f"xgboost_{target_antibiotic}_final_v2.json"
            if v2_path.exists():
                logger.warning(f"Fallback: Model not found at expected path. Using {v2_path.name}")
                model_path = v2_path
            else:
                logger.critical(f"Model not found in {models_dir}")
                sys.exit(1)
            
    logger.info(f"1. Loading model: {model_path.name}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 2. Key Selection (Gain)
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    logger.info(f"2. Identified top {top_n} features.")
    
    # 3. Map back to K-mers
    features_map = {} 
    needed_indices = set()
    for feat_name, score in sorted_importance:
        # XGBoost features are "f123"
        idx = int(feat_name.replace('f', ''))
        needed_indices.add(idx)
        
    logger.info(f"3. Retrieving {len(needed_indices)} DNA sequences from dictionary...")
    
    features_file = matrix_dir / "features.txt"
    if not features_file.exists():
        logger.critical("Features.txt not found! Cannot map features.")
        sys.exit(1)

    with open(features_file, 'r') as f:
        for i, line in enumerate(f):
            if i in needed_indices:
                kmer = line.split()[0]
                features_map[i] = kmer
                
                if len(features_map) == len(needed_indices):
                    break
    
    # 4. Save Results
    results = []
    fasta_lines = []
    
    for rank, (feat_name, score) in enumerate(sorted_importance, 1):
        idx = int(feat_name.replace('f', ''))
        kmer_seq = features_map.get(idx, "UNKNOWN")
        
        results.append({
            'Rank': rank,
            'Feature_ID': feat_name,
            'Gain_Score': score,
            'Kmer_Sequence': kmer_seq
        })
        
        fasta_lines.append(f">Rank_{rank}_Score_{float(score):.2f}\n{kmer_seq}")
        
    df = pd.DataFrame(results)
    csv_path = output_dir / f"top_{top_n}_features_{target_antibiotic}.csv"
    df.to_csv(csv_path, index=False)
    
    fasta_path = output_dir / f"top_{top_n}_features_{target_antibiotic}.fasta"
    with open(fasta_path, 'w') as f:
        f.write("\n".join(fasta_lines))
        
    logger.info("\n--- RESULTS (TOP 10) ---")
    logger.info(f"\n{df.head(10)}")
    logger.info(f"\nSaved to:\n -> {csv_path}\n -> {fasta_path}")

if __name__ == "__main__":
    main()
