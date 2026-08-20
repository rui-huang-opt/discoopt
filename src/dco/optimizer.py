from logging import getLogger
from abc import ABCMeta, abstractmethod
from typing import Callable, cast

import jax
import numpy as np
from numpy.typing import NDArray

from .network import Network
from .regularizer import Regularizer

logger = getLogger("dco.optimizer")

LossFn = Callable[[NDArray[np.float64]], float | NDArray[np.float64] | jax.Array]

ProxG = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


def grad_fn(loss_fn: LossFn) -> Callable[[NDArray[np.float64]], NDArray]:
    """
    Returns a gradient function for the given function loss_fn using either JAX.

    Parameters
    ----------
    loss_fn : Callable[[NDArray[float64]], float64 | Array]
        The function for which to compute the gradient.

    Returns
    -------
    Callable[[NDArray[float64]], NDArray[float64]]
        A function that computes the gradient of loss_fn.
    """

    jax.config.update("jax_platforms", "cpu")

    raw_grad = jax.jit(jax.grad(loss_fn))

    def wrapped_grad(x: NDArray[np.float64]) -> NDArray:
        return np.asarray(raw_grad(x))

    return wrapped_grad


class Optimizer(metaclass=ABCMeta):
    """
    Base class for all optimizers.

    Parameters
    ----------
    loss_fn : Callable[[NDArray[float64]], NDArray[float64] | Array]
        The local loss function to be minimized.

    network : Network
        The network object for communication between nodes.

    gamma : float
        Step size for the optimization algorithm.

    reg : Regularizer | None, optional
        Regularizer to be applied to the optimization problem. If None, no regularization is applied. Default is None.

    Notes
    -----
    This is an abstract base class. Specific optimization algorithms should inherit from this class and implement the `init` and `step` methods.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        self._grad_fn = grad_fn(loss_fn)
        self._network = network
        self._gamma = gamma
        self._aux_var: dict[str, NDArray[np.float64]] = {}

        if reg is None:
            prox_g = lambda tau, x: x
        else:
            prox_g = reg.prox

        self._prox_g = cast(ProxG, prox_g)

    @abstractmethod
    def init(self, x_i: NDArray[np.float64]) -> None:
        """
        Initialize the optimizer's auxiliary variables.

        Args:
            x_i (NDArray[float64]): The initial local variable.
        """
        ...

    @abstractmethod
    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform a single optimization step.

        Args:
            x_i (NDArray[float64]): The current local variable.

        Returns:
            NDArray[float64]: The updated local variable after the optimization step.
        """
        ...


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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        if reg is not None:
            err_msg = "DGD only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        super().__init__(loss_fn, network, gamma, reg)
        self._k = 0

    def init(self, x_i: NDArray[np.float64]) -> None:
        pass

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        w_x_i = self._network.weighted_mix(x_i)
        gamma_bar = self._gamma / np.sqrt(self._k + 1)
        grad = self._grad_fn(x_i)

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        w_x_i = self._network.weighted_mix(x_i)
        self._aux_var["grad"] = self._grad_fn(x_i)
        self._aux_var["new_z_i"] = w_x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        new_x_i = self._prox_g(self._gamma, self._aux_var["new_z_i"])
        p_i = self._aux_var["new_z_i"] + new_x_i - x_i

        w_p_i = self._network.weighted_mix(p_i)
        new_grad = self._grad_fn(new_x_i)
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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_fn(x_i)
        self._aux_var["new_z_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        new_x_i = self._prox_g(self._gamma, self._aux_var["new_z_i"])
        new_s_i = new_x_i - self._gamma * self._grad_fn(new_x_i)

        p_i = self._aux_var["new_z_i"] + new_s_i - self._aux_var["s_i"]

        w_p_i = self._network.weighted_mix(p_i)

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["grad"] = self._grad_fn(x_i)
        self._aux_var["new_z_i"] = x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        new_x_i = self._prox_g(self._gamma, self._aux_var["new_z_i"])
        new_grad = self._grad_fn(new_x_i)
        p_i = new_x_i - x_i - self._aux_var["new_z_i"]

        w_p_i = self._network.weighted_mix(p_i)

        q_i = w_p_i + self._aux_var["new_z_i"] * 2

        w_q_i = self._network.weighted_mix(q_i)

        diff_grad = new_grad - self._aux_var["grad"]
        new_new_z_i = w_q_i - self._gamma * diff_grad

        self._aux_var["grad"] = new_grad
        self._aux_var["new_z_i"] = new_new_z_i

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_fn(x_i)
        self._aux_var["y_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        new_z_i = self._network.weighted_mix(self._aux_var["y_i"])

        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_fn(new_x_i)

        p_i = new_s_i - self._aux_var["s_i"] - new_z_i

        w_p_i = self._network.weighted_mix(p_i)

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["y_i"] = np.zeros_like(x_i)

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._network.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._grad_fn(x_i) - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)

        w_x_i = self._network.weighted_mix(x_i)

        new_y_i = self._aux_var["y_i"] - w_x_i + new_x_i

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["y_i"] = np.zeros_like(x_i)

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._network.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._grad_fn(x_i) - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)

        w_x_i = self._network.weighted_mix(x_i)

        new_y_i = self._aux_var["y_i"] - w_x_i + x_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RAugDGM(Optimizer):
    """
    Robust Augmented Distributed Gradient Method (RAugDGM) algorithm.

    This algorithm is proposed in this paper as a robustified version of AugDGM and a Adaptive-Then-Combine (ATC) variant of RGT.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["y_i"] = np.zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_fn(x_i)

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._network.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_fn(new_x_i)

        q_i = new_x_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._network.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_x_i

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
        loss_fn: LossFn,
        network: Network,
        gamma: float,
        reg: Regularizer | None = None,
    ):
        super().__init__(loss_fn, network, gamma, reg)

    def init(self, x_i: NDArray[np.float64]) -> None:
        self._aux_var["y_i"] = np.zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._grad_fn(x_i)

    def step(self, x_i: NDArray[np.float64]) -> NDArray[np.float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._network.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._grad_fn(new_x_i)

        q_i = new_x_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._network.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


__all__ = [
    "Optimizer",
    "DGD",
    "EXTRA",
    "NIDS",
    "DIGing",
    "AugDGM",
    "RGT",
    "WE",
    "RAugDGM",
    "AtcWE",
]
