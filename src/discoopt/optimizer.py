from logging import getLogger

logger = getLogger("dco.optimizer")

from abc import ABCMeta, abstractmethod
from numpy import sqrt, float64, zeros_like
from numpy.typing import NDArray
from .network import NetworkOps
from .loss_function import LossFunction


class Optimizer(metaclass=ABCMeta):
    """
    Base class for all optimizers.

    Parameters
    ----------
    loss_fn : LossFunction
        The loss function to be minimized.

    ops : NetworkOps
        The network operations for distributed communication.

    gamma : float
        Step size or learning rate for the optimizer.

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

    _SUBCLASSES: dict[str, type["Optimizer"]] = {}

    def __init_subclass__(cls, key: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._SUBCLASSES[key] = cls

    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        self._loss_fn = loss_fn
        self._ops = ops
        self._gamma = gamma
        self._aux_var: dict[str, NDArray[float64]] = {}

    @classmethod
    def create(
        cls, loss_fn: LossFunction, ops: NetworkOps, gamma: float, key: str = "RAugDGM"
    ) -> "Optimizer":
        Subclass = cls._SUBCLASSES.get(key)
        if Subclass is None:
            raise ValueError(f"Algorithm '{key}' is not registered.")
        return Subclass(loss_fn, ops, gamma)

    @abstractmethod
    def init(self, x_i: NDArray[float64]) -> None:
        """Initialize the optimizer's auxiliary variables."""
        ...

    @abstractmethod
    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        """Perform a single optimization step and return the updated variable."""
        ...


class DGD(Optimizer, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        if loss_fn.g_type != "zero":
            err_msg = "DGD only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        super().__init__(loss_fn, ops, gamma)
        self._k = 0

    def init(self, x_i: NDArray[float64]) -> None:
        pass

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        w_x_i = self._ops.weighted_mix(x_i)
        gamma_bar = self._gamma / sqrt(self._k + 1)
        grad = self._loss_fn.grad(x_i)

        new_x_i = w_x_i - gamma_bar * grad
        self._k += 1

        return new_x_i


class EXTRA(Optimizer, key="EXTRA"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        w_x_i = self._ops.weighted_mix(x_i)
        self._aux_var["grad"] = self._loss_fn.grad(x_i)
        self._aux_var["new_z_i"] = w_x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_x_i = self._loss_fn.prox(self._gamma, self._aux_var["new_z_i"])
        p_i = self._aux_var["new_z_i"] + new_x_i - x_i

        w_p_i = self._ops.weighted_mix(p_i)
        new_grad = self._loss_fn.grad(new_x_i)

        diff_grad = new_grad - self._aux_var["grad"]

        new_new_z_i = 0.5 * (p_i + w_p_i) - self._gamma * diff_grad

        self._aux_var["grad"] = new_grad
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class NIDS(Optimizer, key="NIDS"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._loss_fn.grad(x_i)
        self._aux_var["new_z_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_x_i = self._loss_fn.prox(self._gamma, self._aux_var["new_z_i"])
        new_s_i = new_x_i - self._gamma * self._loss_fn.grad(new_x_i)

        p_i = self._aux_var["new_z_i"] + new_s_i - self._aux_var["s_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_new_z_i = 0.5 * (p_i + w_p_i)

        self._aux_var["s_i"] = new_s_i
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class DIGing(Optimizer, key="DIGing"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        if loss_fn.g_type != "zero":
            err_msg = "DIGing only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["grad"] = self._loss_fn.grad(x_i)
        self._aux_var["y_i"] = self._aux_var["grad"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        w_x_i = self._ops.weighted_mix(x_i)

        new_x_i = w_x_i - self._gamma * self._aux_var["y_i"]
        new_grad = self._loss_fn.grad(new_x_i)

        w_y_i = self._ops.weighted_mix(self._aux_var["y_i"])

        new_y_i = w_y_i + new_grad - self._aux_var["grad"]

        self._aux_var["grad"] = new_grad
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AugDGM(Optimizer, key="AugDGM"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["s_i"] = x_i - self._gamma * self._loss_fn.grad(x_i)
        self._aux_var["y_i"] = self._aux_var["s_i"].copy()

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        new_z_i = self._ops.weighted_mix(self._aux_var["y_i"])

        new_x_i = self._loss_fn.prox(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._loss_fn.grad(new_x_i)

        diff_s_i = new_s_i - self._aux_var["s_i"]
        p_i = diff_s_i - new_z_i

        w_p_i = self._ops.weighted_mix(p_i)

        new_y_i = w_p_i + new_z_i * 2

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RGT(Optimizer, key="RGT"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._loss_fn.grad(x_i) - self._aux_var["y_i"]
        new_x_i = self._loss_fn.prox(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class WE(Optimizer, key="WE"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * self._loss_fn.grad(x_i) - self._aux_var["y_i"]
        new_x_i = self._loss_fn.prox(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RAugDGM(Optimizer, key="RAugDGM"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._loss_fn.grad(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._loss_fn.prox(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._loss_fn.grad(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AtcWE(Optimizer, key="AtcWE"):
    def __init__(self, loss_fn: LossFunction, ops: NetworkOps, gamma: float):
        super().__init__(loss_fn, ops, gamma)

    def init(self, x_i: NDArray[float64]) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * self._loss_fn.grad(x_i)

    def step(self, x_i: NDArray[float64]) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._ops.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = self._loss_fn.prox(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._loss_fn.grad(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._ops.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i
