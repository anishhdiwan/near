import argparse
from collections import deque
import torch

def dict2namespace(config):
    """Convert a disctionary (typically containing config params) to a namespace structure (https://tedboy.github.io/python_stdlib/generated/generated/argparse.Namespace.html#argparse.Namespace)

    Args:
        config (dict): dictionary of configs params
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class LastKMovingAvg:
    """Given a time series where data is continously being added, keep a buffer of the last k elements and return their stats

    Assumes that the elements are torch tensors
    """

    def __init__(self, maxlen=3):
        self.maxlen = maxlen
        self.buffer = deque([], maxlen=self.maxlen)
    
    def append(self, element, return_avg=True):
        self.buffer.append(element)

        if return_avg:
            return torch.mean(torch.stack(list(self.buffer))).item()

    def reset(self):
        self.buffer = deque([], maxlen=self.maxlen)


