# downstream/__init__.py

from .linear_probe import train_linear_probe
from .knn_probe import train_knn_probe

# allowed list of kwargs for each probe
_LINEAR_KW = {'solver', 'penalty', 'C', 'class_weight',
              'multi_class', 'n_jobs', 'warm_start', 'verbose',
              'weight_decay', 'max_iter', 'task_type', 'X_val', 
              'y_val', 'X_test', 'y_test'}

_KNN_KW = {'n_neighbors', 'weights', 'algorithm', 'leaf_size',
           'metric', 'p', 'task_type', 'X_val', 'y_val', 
           'X_test', 'y_test'}

def run_probe(*, kind: str, features, targets, **kwargs):
    # Parameters
    # Kind: {'linear', 'knn'} as of right now
    # All other kwargs are forwaded to underlying trainer
    kind = kind.lower()
    if kind == 'linear':
        kw = {k: v for k, v in kwargs.items() if k in _LINEAR_KW}
        return train_linear_probe(features, targets, **kw)
    elif kind == 'knn':
        kw = {k: v for k, v in kwargs.items() if k in _KNN_KW}
        return train_knn_probe(features, targets, **kw)
    else:
        raise ValueError(f"Unknown probe kind '{kind}'")