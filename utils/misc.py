# utils/misc.py

import numpy as np
import pandas as pd

def to_f32(a):
    # accept ndarray / DataFrame / Series / list
    # returns contiguous float32 array for pytorch
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.to_numpy(dtype = np.float32, copy = False)
    return np.asarray(a, dtype = np.float32)