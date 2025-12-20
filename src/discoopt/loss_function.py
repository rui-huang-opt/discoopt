from typing import Callable, Any
from numpy import float64
from numpy.typing import NDArray
from .regularizer import Regularizer


def grad(
    f: Callable[[NDArray[float64]], Any], backend: str
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    if backend == "jax":
        from jax import grad, jit, config, device_get

        config.update("jax_platforms", "cpu")

        raw_grad = jit(grad(f))

        def wrapped_grad(x: NDArray[float64]) -> NDArray[float64]:
            return device_get(raw_grad(x))

        return wrapped_grad

    elif backend == "autograd":
        from autograd import grad

        return grad(f)  # type: ignore

    else:
        raise ValueError(f"Unsupported backend: {backend}")


class LossFunction:
    """
    Represents a loss function composed of a differentiable part and a regularizer.

    Parameters
    ----------
    f_i : Callable[[NDArray[float64]], float | Any]
        The differentiable part of the loss function.

    g_type : str, optional
        The type of regularizer to use (default is "zero", which means no regularization).

    lam : float, optional
        The regularization parameter (default is 1.0).

    backend : str, optional
        The backend to use for automatic differentiation (default is "autograd").
    """

    def __init__(
        self,
        f_i: Callable[[NDArray[float64]], float | Any],
        g_type: str = "zero",
        lam: float = 1.0,
        backend: str = "autograd",
    ):
        self._f_i = f_i
        self._g_type = g_type
        self._g = Regularizer.create(g_type, lam)
        self.grad = grad(f_i, backend)
        self.prox = self._g.prox

    @property
    def g_type(self) -> str:
        return self._g_type

    def __call__(self, x: NDArray[float64]) -> float:
        return float(self._f_i(x) + self._g(x))
