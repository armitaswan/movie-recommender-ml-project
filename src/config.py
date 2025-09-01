from pathlib import Path

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROC_DIR = DATA_DIR / 'processed'
MODELS_DIR = Path('models')
SEED = 42
TOP_K = 10
ALPHA = 0.6
MIN_VOTES_PERCENTILE = 0.80
