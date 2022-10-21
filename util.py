import math

import numpy as np


def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}


def log_factorial(x: int) -> float:
    return math.lgamma(x) + np.log(x)
