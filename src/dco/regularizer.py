import numpy as np
from typing import Protocol
from numpy.linalg import norm
from numpy.typing import NDArray


class Regularizer(Protocol):
    """
    Protocol for regularizer functions used in optimization.
    Includes methods for evaluating the regularizer and computing its proximal operator.
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the regularizer at the given point.

        Args:
            x (NDArray[np.float64]): The input array.

        Returns:
            float: The value of the regularizer at x.
        """
        ...

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the proximal operator of the regularizer.

        Args:
            tau (float): The step size parameter.
            x (NDArray[np.float64]): The input array.

        Returns:
            NDArray[np.float64]: The result of the proximal operator.
        """
        ...


class L1:
    """
    L1 regularizer.

    Parameters
    ----------
    lam : float
        Regularization parameter.

    Notes
    -----
    Parameters lam and tau have different roles:
    - lam controls the strength of the regularization in the objective function.
    - tau is a step size parameter used in the proximal operator computation.
    However, both parameters influence the amount of shrinkage applied to the input x
    """

    def __init__(self, lam: float):
        self._lam = lam

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (self._lam * norm(x, ord=1)).item()

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        threshold = tau * self._lam
        return np.multiply(np.sign(x), np.maximum(np.abs(x) - threshold, 0))


class WeightedL1:
    """
    Weighted L1 regularizer.

    Parameters
    ----------
    weights : NDArray[np.float64]
        Weights for each component of the input vector.
    """

    def __init__(self, weights: NDArray[np.float64]):
        self._weights = weights

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (np.sum(self._weights * np.abs(x))).item()

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        threshold = tau * self._weights
        return np.multiply(np.sign(x), np.maximum(np.abs(x) - threshold, 0))


class L2:
    """
    L2 regularizer.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    """

    def __init__(self, lam: float):
        self._lam = lam

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (self._lam * norm(x, ord=2)).item()

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_norm = norm(x, ord=2)
        if x_norm == 0:
            return x
        factor = np.maximum(0, 1 - (tau * self._lam) / x_norm)
        return factor * x


class SquaredL2:
    """
    Squared L2 regularizer.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    """

    def __init__(self, lam: float):
        self._lam = lam

    def __call__(self, x: NDArray[np.float64]) -> float:
        return (0.5 * self._lam * norm(x, ord=2) ** 2).item()

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        factor = 1 / (1 + tau * self._lam)
        return factor * x


class Box:
    """
    Box constraint regularizer.

    Parameters
    ----------
    lower_bound : float
        Lower bound of the box.

    upper_bound : float
        Upper bound of the box.
    """

    def __init__(self, lower_bound: float = -1.0, upper_bound: float = 1.0):
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


class NonNegative:
    """
    Non-negativity constraint regularizer.

    Notes
    -----
    This regularizer enforces that all components of the input vector are non-negative,
    which is useful in dual formulations of optimization problems with inequality constraints.
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        if np.all(x >= 0):
            return 0.0
        else:
            return np.inf

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(x, 0)


__all__ = ["L1", "WeightedL1", "L2", "SquaredL2", "Box", "NonNegative"]
