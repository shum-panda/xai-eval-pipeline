import functools

import torch


def with_cuda_cleanup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return wrapper