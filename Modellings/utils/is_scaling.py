import numpy as np
from functools import reduce

def _is_scaling(data):
    mean_sum = np.mean(data, axis=0).sum()
    std_multiply = reduce(lambda x, y: x * y, np.std(data, axis=0))

    mean_cond = mean_sum < 1e-6
    std_cond = abs(std_multiply - 1.0) < 1e-6
    
    if all([mean_cond, std_cond]):
        return True
    else:
        return False
