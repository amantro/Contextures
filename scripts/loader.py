# scripts/loader.py

"""
Data access layer for OpenML datasets

Usage: 
X, y, meta = load_dataset('openml__iris__59')
"""

from __future__ import annotations
import gzip, json
from pathlib import Path
from typing import Tuple, Dict

from scripts.openml_split import fetch_data_and_splits

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / 'data'

def load_dataset(tag: str, *, as_numpy: bool = False) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    # as_numpy = True - np.ndarray else pd.DataFrame
    root = DATA_DIR / tag
    if not root.exists():
        raise FileNotFoundError(f'Dataset {tag} not found - run scripts/download_openml.py')
    
    with gzip.open(root / 'X.npy.gz', 'rb') as f:
        X = np.load(f, allow_pickle = True)
    with gzip.open(root / 'y.npy.gz', 'rb') as f:
        y = np.load(f, allow_pickle = True)
    
    with open(root / 'metadata.json') as f:
        meta = json.load(f)
    
    if as_numpy:
        return X, y, meta
    
    feat_names = meta.get('feature_names', [f'f{i}' for i in range(X.shape[1])])
    df = pd.DataFrame(X, columns = feat_names)

    cat_idx = set(meta.get('cat_idx', []))
    for j in cat_idx:
        df.iloc[:, j] = df.iloc[:, j].astype('category')
    
    return df, y, meta

def load_openml_dataset(task_id: int, *, val_ratio: float = 0.1, random_state: int = 0, as_numpy: bool = False):
    # returns (X, y, meta, split_list)
    # cached under data/openml_cache/ so every parallel job shares one download
    cache = DATA_DIR / 'openml_cache'
    X, y, meta, splits = fetch_data_and_splits(task_id,
                                                   cache_dir = cache,
                                                   val_ratio = val_ratio,
                                                   random_state = random_state)
    
    return (X.values if as_numpy else X), y, meta, splits