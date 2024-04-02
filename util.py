import math

import numpy as np


class Token:
    def __init__(self, token: str, is_start_of_word: bool):
        self.token = token
        self.is_start_of_word = is_start_of_word

    def __repr__(self):
        return ('' if self.is_start_of_word else '#') + self.token

    def __eq__(self, other):
        return isinstance(other, Token) and self.token == other.token and self.is_start_of_word == other.is_start_of_word


def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}


def log_factorial(x: int) -> float:
    return math.lgamma(x) + np.log(x)
