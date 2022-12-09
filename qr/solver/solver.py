from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import warnings

class AbstractQRSolver(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        pass
    @abstractmethod
    def solve(self, A: np.ndarray) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        pass

class SolverFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class: AbstractQRSolver) -> Callable:
            if name in cls.registry:
                warnings.warn(f'Register {name} already exists. Will replace it')
            cls.registry[name] = wrapped_class
        return inner_wrapper

    @classmethod
    def create_solver(cls, name: str, **kwargs):
        exec_class = cls.registry[name]
        executer = exec_class(**kwargs)
        return executer
