from logging import getLogger

logger = getLogger("dco.optimizer")

from abc import ABCMeta, abstractmethod
from typing import Callable, Type
from numpy import sqrt, float64, zeros_like
from numpy.typing import NDArray
from jax import Array
from ._autodiff import nabla
from ._network import NetworkOps
from .regularizer import Regularizer


class Optimizer(metaclass=ABCMeta):
    """
    Base class for all optimizers.

    Parameters
    ----------
    loss_fn : Callable[[NDArray[float64]], NDArray[float64] | Array]
        The local loss function to be minimized.

    ops : NetworkOps
        Network operations for communication between nodes.

    gamma : float
        Step size for the optimization algorithm.

    reg : Regularizer | None, optional
        Regularizer to be applied to the optimization problem. If None, no regularization is applied. Default is None.

    use_jax : bool, optional
        Whether to use JAX for automatic differentiation. If False, Autograd will be used. Default is True.

    Methods
    -------
    init(x_i: NDArray[float64]) -> None
        Initialize the optimizer's auxiliary variables.

    step(x_i: NDArray[float64]) -> NDArray[float64]
        Perform a single optimization step and return the updated variable.

    Notes
    -----
    This is an abstract base class. Specific optimization algorithms should inherit from this class and implement the `init` and `step` methods.
    """

    _registry: dict[str, Type["Optimizer"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        self._grad_f = nabla(loss_fn, use_jax)
        self._prox_g: Callable[[float, NDArray[float64]], NDArray[float64]]
        self._ops = ops
        self._gamma = gamma
        self._aux_var: dict[str, NDArray[float64]] = {}

        if reg is None:
            self._prox_g = lambda tau, x: x
        else:
            self._prox_g = reg.prox

    @abstractmethod
    def init(self, x_i: NDArray[float64]) -> None: ...

    @abstractmethod
    def step(self, x_i: NDArray[float64]) -> NDArray[float64]: ...

    @classmethod
    def get_class(cls, name: str) -> Type["Optimizer"]:
        """
        Get the optimizer class by name.
        This is not a recommended way to create optimizers.
        Use the specific optimizer class directly instead.
        Here is provided for convenience for benchmarking and testing purposes.

        Parameters
        ----------
        name : str
            Name of the optimizer class.

        Returns
        -------
        Type[Optimizer]
            The optimizer class corresponding to the given name.

        Raises
        ------
        ValueError
            If the optimizer class with the given name is not found.
        """
        if name not in cls._registry:
            err_msg = f"Optimizer class '{name}' not found."
            logger.error(err_msg)
            raise ValueError(err_msg)

        return cls._registry[name]


class DGD(Optimizer):
    """
    Distributed Gradient Descent (DGD) algorithm.

    The paper introducing the DGD algorithm is:
    Nedic, A., & Ozdaglar, A. (2009).
    "Distributed Subgradient Methods for Multi-Agent Optimization".
    IEEE Transactions on Automatic Control, 54(1), 48-61.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        if reg is not None:
            err_msg = "DGD only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        super().__init__(loss_fn, ops, gamma, reg, use_jax)
        self._k = 0

    def init(self, x_i: NDArray[float64]) -> None:
        pass

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        w_x_i = self._ops.weighted_mix(x_i)
        gamma_bar = self._gamma / sqrt(self._k + 1)
        grad = self._grad_f(x_i)

        new_x_i = w_x_i - gamma_bar * grad
        self._k += 1

        return new_x_i


class EXTRA(Optimizer):
    """
    Exact First-Order Algorithm (EXTRA).

    The paper introducing the EXTRA algorithm is:
    Shi, W., Ling, Q., Wu, G., & Yin, W. (2015).
    "EXTRA: An Exact First-Order Algorithm for Decentralized Consensus Optimization".
    SIAM Journal on Optimization, 25(2), 944-966.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        w_x_i = self._ops.weighted_mix(x_i)
        self._aux_var["grad"] = self._grad_f(x_i)
        self._aux_var["new_z_i"] = w_x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_x_i = self._prox_g(self._gamma, self._aux_var["new_z_i"])
        p_i = self._aux_var["new_z_i"] + new_x_i - x_i

        w_p_i = self._ops.weighted_mix(p_i)
        new_grad = self._grad_f(new_x_i)
        diff_grad = new_grad - self._aux_var["grad"]

        new_new_z_i = 0.5 * (p_i + w_p_i) - self._gamma * diff_grad

        self._aux_var["grad"] = new_grad
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class NIDS(Optimizer):
    """
    Network Independent Step-size (NIDS) algorithm.

    The paper introducing the NIDS algorithm is:
    Li, Z., Shi, W., & Yan, M. (2019).
    "A decentralized proximal-gradient method with network independent step-sizes and separated convergence rates".
    IEEE Transactions on Signal Processing, 67(17), 4494-4506.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_f(x_i)
        self._aux_var["new_z_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_x_i = self._prox_g(self._gamma, self._aux_var["new_z_i"])
        new_s_i = new_x_i - self._gamma * self._grad_f(new_x_i)

        p_i = self._aux_var["new_z_i"] + new_s_i - self._aux_var["s_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_new_z_i = 0.5 * (p_i + w_p_i)

        self._aux_var["s_i"] = new_s_i
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class DIGing(Optimizer):
    """
    Distributed Inexact Gradient and a gradient tracking algorithm (DIGing).

    The paper introducing the DIGing algorithm is:
    Nedic, A., Olshevsky, A., & Shi, W. (2017).
    "Achieving Geometric Convergence for Distributed Optimization over Time-Varying Graphs".
    SIAM Journal on Optimization, 27(4), 2597-2633.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        if reg is not None:
            err_msg = "DIGing only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["grad"] = self._grad_f(x_i)
        self._aux_var["y_i"] = self._aux_var["grad"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        w_x_i = self._ops.weighted_mix(x_i)

        new_x_i = w_x_i - self._gamma * self._aux_var["y_i"]
        new_grad = self._grad_f(new_x_i)

        w_y_i = self._ops.weighted_mix(self._aux_var["y_i"])

        new_y_i = w_y_i + new_grad - self._aux_var["grad"]

        self._aux_var["grad"] = new_grad
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AugDGM(Optimizer):
    """
    Augmented Distributed Gradient Method (AugDGM) algorithm.

    To make the algorithm compliant with the Adaptive-Then-Combine (ATC) framework,
    where only the post-descent variable is required at each iteration,
    we reformulate AugDGM into an equivalent form that differs from the one presented in the original paper.

    The paper introducing the AugDGM algorithm is:
    Xu, J., Zhu, S., Soh, Y. C., & Xie, L. (2015).
    "Augmented Distributed Gradient Methods for Multi-Agent Optimization under Uncoordinated Constant Stepsizes".
    In Proceedings of the 54th IEEE Conference on Decision and Control (CDC) (pp. 2055-2060). IEEE.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_f(x_i)
        self._aux_var["y_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_z_i = self._ops.weighted_mix(self._aux_var["y_i"])

        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_f(new_x_i)

        diff_s_i = new_s_i - self._aux_var["s_i"]
        p_i = diff_s_i - new_z_i

        w_p_i = self._ops.weighted_mix(p_i)

        new_y_i = w_p_i + new_z_i * 2

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RGT(Optimizer):
    """
    Robust Gradient Tracking (RGT) algorithm.

    The paper introducing the RGT algorithm is:
    Pu, S. (2020).
    "A robust gradient tracking method for distributed optimization over directed networks".
    In Proceedings of the 59th IEEE Conference on Decision and Control (CDC) (pp. 2335-2341). IEEE.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._grad_f(x_i) - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class WE(Optimizer):
    """
    Wang-Elia (WE) algorithm.

    The paper introducing the WE algorithm is:
    Wang, J., & Elia, N. (2010).
    "Control approach to distributed optimization".
    In Proceedings of the 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton) (pp. 557-561). IEEE.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._grad_f(x_i) - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RAugDGM(Optimizer):
    """
    Robust Augmented Distributed Gradient Method (RAugDGM) algorithm.

    This algorithm is proposed in this paper as a robustified version of AugDGM and a Adaptive-Then-Combine (ATC) variant of RGT.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_f(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_f(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AtcWE(Optimizer):
    """
    Adaptive-Then-Combine Wang-Elia (AtcWE) algorithm.

    This algorithm is proposed in this paper as a Adaptive-Then-Combine (ATC) version of WE.
    """

    def __init__(
        self,
        loss_fn: Callable[[NDArray[float64]], float | NDArray[float64] | Array],
        ops: NetworkOps,
        gamma: float,
        reg: Regularizer | None = None,
        use_jax: bool = True,
    ):
        super().__init__(loss_fn, ops, gamma, reg, use_jax)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_f(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_f(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


__all__ = ["DGD", "EXTRA", "NIDS", "DIGing", "AugDGM", "RGT", "WE", "RAugDGM", "AtcWE"]
