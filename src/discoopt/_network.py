from typing import Protocol, KeysView
from numpy import float64
from numpy.typing import NDArray


class NetworkOps(Protocol):
    """
    Protocol for communication operations in distributed optimization.
    This protocol must realize the weighted mixing operation used in distributed optimization algorithms.
    It assumes the existence of a weight matrix W, where W_ij represents the weight from node i to neighbor j.

    In our examples, we use 'topolink.NodeHandle' as a concrete implementation of this protocol.
    The source code of 'topolink.NodeHandle' can be found at:
        https://github.com/rui-huang-opt/topolink.

    However, any class that implements these methods and properties can be used as long as it adheres to this protocol.
    """

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        ...
