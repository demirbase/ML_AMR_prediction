import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
import sys
import gc

# Add project root to sys.path to import src
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import load_config, setup_logger

def run_command(cmd, logger):
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        sys.exit(1)

def main():
    # Setup
    config = load_config()
    logger = setup_logger("matrix_generation", log_file=config['paths']['logs_dir'] / "02_matrix_generation.log")
    
    logger.info("--- PARTIAL MATRIX GENERATION (RAM OPTIMIZED) ---")
    
    # Config parameters
    target_antibiotic = config['project']['target_antibiotic']
    min_support = config['preprocessing']['min_support']
    kmc_mem = config['preprocessing']['kmc_mem']
    threads = config['preprocessing']['threads']
    chunk_size = config['preprocessing']['chunk_size']
    
    # Paths
    raw_genomes_dir = Path(config['paths']['raw_genomes_dir'])
    kmc_outputs_dir = Path(config['paths']['kmc_outputs_dir'])
    matrix_out_dir = Path(config['paths']['matrix_dir'])
    temp_dir = kmc_outputs_dir / "tmp"
    metadata_file = Path(config['paths']['metadata_file'])
    
    # KMC Binaries - expand user if needed (handled by Config? No, simplistic string in config)
    # The config loader handles relative paths to project root, but ~/bin is external.
    # We'll expect config to provide full path or expand here.
    kmc_bin = config['preprocessing']['kmc_bin']
    if kmc_bin.startswith("~"):
        kmc_bin = Path(kmc_bin).expanduser()
    
    kmc_tools_bin = config['preprocessing']['kmc_tools_bin']
    if kmc_tools_bin.startswith("~"):
        kmc_tools_bin = Path(kmc_tools_bin).expanduser()

    # Create directories
    matrix_out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    kmc_outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata and File Check
    logger.info("1. Preparing metadata...")
    if not metadata_file.exists():
        logger.critical(f"Metadata file not found: {metadata_file}")
        sys.exit(1)
        
    df = pd.read_csv(metadata_file)
    df['Genome ID'] = df['Genome ID'].astype(str)
    
    # Filter for target antibiotic
    if target_antibiotic not in df.columns:
        logger.critical(f"Target antibiotic '{target_antibiotic}' not in metadata columns.")
        sys.exit(1)
        
    df_target = df.dropna(subset=[target_antibiotic])
    
    all_ids = df_target['Genome ID'].values
    all_labels = df_target[target_antibiotic].astype(int).values
    
    valid_genomes = []
    valid_labels = []
    
    # Check for KMC output existence
    # Note: Original script checked for {gid}.kmc_pre in KMC_OUTPUTS_DIR
    for i, gid in enumerate(all_ids):
        if (kmc_outputs_dir / f"{gid}.kmc_pre").exists():
            valid_genomes.append(gid)
            valid_labels.append(all_labels[i])
            
    logger.info(f"   -> Total Genomes to Process: {len(valid_genomes)}")
    if len(valid_genomes) == 0:
        logger.warning("No valid genomes found with KMC outputs. Check preprocessing.")
        sys.exit(0)

    # 2. GLOBAL DICTIONARY (FEATURE LIST) CREATION
    features_file = matrix_out_dir / "features.txt"
    
    if features_file.exists():
        logger.info("2. Global dictionary already exists, loading...")
    else:
        logger.info("2. Creating global dictionary (Scanning all genomes)...")
        global_list_file = temp_dir / "global_input_list.txt"
        with open(global_list_file, 'w') as f:
            for gid in valid_genomes:
                # Assuming raw genomes are .fna
                f.write(str(raw_genomes_dir / f"{gid}.fna") + "\n")
        
        global_db = temp_dir / "global_features_db"
        # -ci{MIN_SUPPORT} eliminates rare kmers
        cmd = f"{kmc_bin} -k31 -m{kmc_mem} -t{threads} -ci{min_support} -fm @{global_list_file} {global_db} {temp_dir}"
        run_command(cmd, logger)
        
        # Dump and transform
        run_command(f"{kmc_tools_bin} transform {global_db} dump {features_file}", logger)
        
        # Cleanup
        run_command(f"rm {global_db}.*", logger)

    # Load Dictionary into RAM
    logger.info("   -> Loading dictionary into RAM...")
    kmer_to_idx = {}
    with open(features_file, 'r') as f:
        for idx, line in enumerate(f):
            kmer, _ = line.split()
            kmer_to_idx[kmer] = idx
    n_features = len(kmer_to_idx)
    logger.info(f"   -> Total Features: {n_features}")

    # 3. CHUNKED PROCESSING
    logger.info(f"\n3. Starting chunked processing (Chunk Size: {chunk_size})...")
    
    # Save labels and IDs
    pd.DataFrame(valid_labels, columns=['label']).to_csv(matrix_out_dir / f"y_{config['training']['file_prefix']}.csv", index=False)
    pd.DataFrame(valid_genomes, columns=['Genome ID']).to_csv(matrix_out_dir / f"genomes_{config['training']['file_prefix']}.csv", index=False)

    total_genomes = len(valid_genomes)
    num_chunks = (total_genomes + chunk_size - 1) // chunk_size
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, total_genomes)
        
        chunk_genomes = valid_genomes[start_idx:end_idx]
        chunk_out_file = matrix_out_dir / f"X_{config['training']['file_prefix']}_part_{chunk_id}.npz"
        
        # Resume capability
        if chunk_out_file.exists():
            logger.info(f"   -> Chunk {chunk_id+1}/{num_chunks} already exists, skipping.")
            continue

        logger.info(f"   -> Processing: Chunk {chunk_id+1}/{num_chunks} ({len(chunk_genomes)} genomes)")
        
        row_ind = []
        col_ind = []
        data = []
        
        for local_i, gid in enumerate(tqdm(chunk_genomes, leave=False)):
            db_path = kmc_outputs_dir / gid
            tmp_dump = temp_dir / f"{gid}_dump.txt"
            
            # Dump KMC db to text
            run_command(f"{kmc_tools_bin} transform {db_path} dump {tmp_dump}", logger)
            
            # Read text and map to global features
            with open(tmp_dump, 'r') as f:
                for line in f:
                    kmer, _ = line.split()
                    if kmer in kmer_to_idx:
                        col_id = kmer_to_idx[kmer]
                        row_ind.append(local_i) 
                        col_ind.append(col_id)
                        data.append(1)
            
            # Remove tmp file
            try:
                os.remove(tmp_dump)
            except OSError:
                pass
        
        # Create CSR Matrix
        X_chunk = csr_matrix((data, (row_ind, col_ind)), shape=(len(chunk_genomes), n_features), dtype=np.int8)
        save_npz(chunk_out_file, X_chunk)
        
        # Cleanup RAM
        del X_chunk, row_ind, col_ind, data
        gc.collect()

    logger.info("\n>>> ALL CHUNKS CREATED! <<<")
    logger.info(f"Files are in {matrix_out_dir}")

if __name__ == "__main__":
    main()
