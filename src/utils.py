import yaml
import logging
from pathlib import Path
import sys

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file and resolves paths to be relative to the project root.
    """
    # Assuming this is called from within the project structure, or relative to where it's run
    # If run from root: config/config.yaml is valid
    # We want to establish the Project Root. 
    # If utils.py is in src/, project root is parent of src.
    
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent # src/ -> AMR_Genomic_Project/
    
    full_config_path = project_root / config_path
    
    if not full_config_path.exists():
        # Fallback: maybe running from root and config is just "config/config.yaml"
        full_config_path = Path(config_path).resolve()
    
    if not full_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {full_config_path}")
    
    with open(full_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Store project root in config for reference
    config["paths"]["project_root"] = project_root

    # Resolve paths in the 'paths' section
    for key, value in config.get("paths", {}).items():
        if key == "project_root": continue
        
        # Handle ~ expansion and convert to Path
        p = Path(value).expanduser()
        
        # If absolute, keep it. If relative, make it relative to project_root
        if not p.is_absolute():
            config["paths"][key] = project_root / p
        else:
            config["paths"][key] = p
            
    return config

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Sets up a logger with the specified name, log file, and level.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
