import json, pickle
from pathlib import Path

def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
