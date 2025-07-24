# scripts/openml_split.py

"""
One place to fetch data *and* official train/test splits

fetch_data_and_splits(task_id, cache_dir, val_ratio, random_state) ->
(X_df, y, meta_dict, list_of_(train, val, test)_index_tuples)
"""

from __future__ import annotations
from pathlib import Path 
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import openml
import openml.config as oml_cfg

# helpers
def _task_to_target_type(task) -> str:
    t = task.task_type.lower() if isinstance(task.task_type, str) else task.task_type.value.lower()
    if 'classification' in t:
        return 'binary' if 'binary' in t else 'classification'
    if 'regression' in t:
        return 'regression'
    raise ValueError(f'Unsupported task type: {task.task_type}')

# public methods
def fetch_data_and_splits(task_id: int, *, cache_dir: Path, val_ratio: float = 0.1, 
                          random_state: int | None = 0) -> Tuple[pd.DataFrame, np.ndarray, Dict,
                                                                 List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    # downloads once, then re-uses OpenML cache
    # Each official split (repeat x fold x sample) is turned into (train_idx, val_idx, test_idx)
    # where *val* is a deterministic 'val_ratio' slice inside the train fold
    cache_dir.mkdir(parents = True, exist_ok = True)
    oml_cfg.cache_directory = str(cache_dir)

    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format = 'dataframe')

    n_rep, n_fold, n_samp = task.get_split_dimensions()
    rng = np.random.RandomState(random_state)

    split_tuples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for r in range(n_rep):
        for f in range(n_fold):
            for s in range(n_samp):
                train_idx, test_idx = task.get_train_test_split_indices(repeat = r,
                                                                        fold = f,
                                                                        sample = s)
                if val_ratio > 0:
                    perm = rng.permutation(train_idx)
                    n_val = int(len(perm) * val_ratio)
                    val_idx, train_idx = perm[:n_val], perm[n_val:]
                else:
                    val_idx = np.array([], dtype = int)
                split_tuples.append((train_idx, val_idx, test_idx))
    
    meta = dict(
        openml_task_id = task_id,
        dataset_id = task.dataset_id,
        target_name = task.target_name,
        n_repeats = n_rep,
        n_folds = n_fold,
        n_samples = n_samp,
        target_type = _task_to_target_type(task),
        feature_names = X.columns.tolist()
    )

    return X, y.to_numpy(), meta, split_tuples
