import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod


class Regularizer(metaclass=ABCMeta):
    _SUBCLASSES: dict[str, type["Regularizer"]] = {}

    def __init_subclass__(cls, key: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._SUBCLASSES[key] = cls

    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = lam

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> float: ...

    @abstractmethod
    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @property
    def lam(self) -> float:
        return self._lam

    @lam.setter
    def lam(self, value: float):
        if value <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = value

    @classmethod
    def create(cls, g_type: str, lam: float, *args, **kwargs):
        Subclass = cls._SUBCLASSES.get(g_type)
        if Subclass is None:
            raise ValueError(f"Regularizer '{g_type}' is not registered.")
        return Subclass(lam, *args, **kwargs)


class Zero(Regularizer, key="zero"):
    def __init__(self, lam: float):
        super().__init__(lam)

    def __call__(self, x: NDArray[np.float64]) -> float:
        return 0

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x


class L1(Regularizer, key="l1"):
    def __init__(self, lam: float):
        super().__init__(lam)

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (self._lam * norm(x, ord=1)).astype(float)

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        threshold = tau * self._lam
        return np.multiply(np.sign(x), np.maximum(np.abs(x) - threshold, 0))


class L2(Regularizer, key="l2"):
    def __init__(self, lam: float):
        super().__init__(lam)

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (0.5 * self._lam * norm(x, ord=2) ** 2).astype(float)

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        factor = 1 / (1 + tau * self._lam)
        return factor * x


class Box(Regularizer, key="box"):
    def __init__(self, lam: float, lower_bound: float = -1.0, upper_bound: float = 1.0):
        super().__init__(lam)
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __call__(self, x: NDArray[np.float64]) -> float:
        if np.all(x >= self._lower_bound) and np.all(x <= self._upper_bound):
            return 0.0
        else:
            return np.inf

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(x, self._lower_bound, self._upper_bound)
