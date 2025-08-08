import functools
import gc
from typing import Any, Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


def with_cuda_cleanup(func: F) -> F:
    """
    Decorator that ensures CUDA memory is cleaned up after the function call.

    This is useful in long-running processes or evaluation loops where
    memory fragmentation or leakage may occur on the GPU.

    The decorated function is executed as usual, and then `torch.cuda.empty_cache()`
    is called in a `finally` block if CUDA is available.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with CUDA cleanup logic.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> F:
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return wrapper
