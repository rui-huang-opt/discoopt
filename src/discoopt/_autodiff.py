from typing import Callable
from numpy import float64
from numpy.typing import NDArray
from jax import Array


def nabla(
    f: Callable[[NDArray[float64]], NDArray[float64] | Array], use_jax: bool
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    """
    Returns a gradient function for the given function f using either JAX or Autograd.

    Parameters
    ----------
    f : Callable[[NDArray[float64]], float64 | Array]
        The function for which to compute the gradient.

    use_jax : bool
        Whether to use JAX for automatic differentiation.
        If False, Autograd will be used, which requires the 'autograd' package to be installed.
        Use 'pip install discoopt[autograd]' to install the package.
        This option is useful when JAX does not work properly (e.g., on Raspberry Pi).

    Returns
    -------
    Callable[[NDArray[float64]], NDArray[float64]]
        A function that computes the gradient of f.
    """

    if not use_jax:
        from autograd import grad

        return grad(f)  # type: ignore

    from jax import grad, jit, config, device_get

    config.update("jax_platforms", "cpu")

    raw_grad = jit(grad(f))

    def wrapped_grad(x: NDArray[float64]) -> NDArray[float64]:
        return device_get(raw_grad(x))

    return wrapped_grad
