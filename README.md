# Distributed Composite Optimization (DisCoOpt)

Distributed Composite Optimization (DisCoOpt) is a Python package for solving distributed composite optimization problems of the form

$$
\min_{x \in \mathbb{R}^d}
\quad
\sum_{i=1}^n
\left(
f_i(x) + g_i(x)
\right),
$$

where each $f_i: \mathbb{R}^d \rightarrow \mathbb{R}$ is a smooth loss function associated with a local dataset or agent, and each $g_i: \mathbb{R}^d \rightarrow \mathbb{R} \cup {+\infty}$ is a possibly nonsmooth regularization term.

DisCoOpt provides implementations of both standard and robust distributed optimization algorithms, making it suitable for federated learning, multi-agent systems, decentralized learning, and large-scale optimization.

DisCoOpt also contains the implementations used in the experiments of the paper [*A Unified Framework for Robust Distributed Optimization under Bounded Disturbances*](https://ieeexplore.ieee.org/abstract/document/11178647).

## Features

* Support for distributed and parallel optimization.
* Standard and robust distributed optimization algorithms.
* Modular and extensible architecture.
* Formula-style API based on NumPy and JAX, allowing distributed optimization algorithms to be defined and deployed across multiple machines as naturally as writing mathematical expressions.
* Distributed network topology and neighbor-to-neighbor communication through [ConOps](https://github.com/rui-huang-opt/conops).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/rui-huang-opt/discoopt.git
```

Or, for development:

```bash
git clone https://github.com/rui-huang-opt/discoopt.git
cd discoopt
pip install -e .
```

## Usage Example

This example demonstrates distributed ridge regression, where each node has a local least-squares objective with $\ell_2$ regularization:

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{4} \sum_{i = 1}^4 (u_i^\top x - v_i)^2 + \rho \| x \|^2.
$$

We use the `EXTRA` algorithm [[1]](#references) to solve this problem:

$$
\mathbf{x}^{k + 2} = (I + W) \mathbf{x}^{k + 1} - \frac{I + W}{2} \mathbf{x}^k - \gamma \big(\nabla f(\mathbf{x}^{k + 1}) - \nabla f(\mathbf{x}^{k})\big),
$$

where $W$ is a symmetric mixing matrix determined by the network topology.

Each node defines its own local data and optimization parameters, including the feature vector `u_i`, target value `v_i`, regularization parameter `rho`, and step size.

The local objective function at node $i$ is

$$
f_i(x_i) = (u_i^\top x_i - v_i)^2 + \rho \|x_i\|^2.
$$

Algorithms are selected explicitly by importing the corresponding optimizer class.

For example:

```python
import numpy as np

from numpy.typing import NDArray

from conops import Network
from dco import EXTRA


# Node-specific data (replace with your own)
node_id = "1"

u_i = np.array([1.0, 2.0, 3.0])
v_i = np.array([1.0])

rho = 0.1
dimension = 3
step_size = 0.01


# Neighbor weights for this node
neighbors: dict[str, float] = {
    "2": 0.2,
    "4": 0.2,
}


def f_i(x_i: NDArray[np.float64]) -> NDArray[np.float64]:
    return (u_i @ x_i - v_i) ** 2 + rho * (x_i @ x_i)


network = Network(node_id, neighbors)

optimizer = EXTRA(f_i, network, step_size)

x_i = np.zeros(dimension)

optimizer.init(x_i)

network.start()

k = network.round_id

while k < 500:
    x_i = optimizer.step(x_i)

    print(f"Step {k}, x_{node_id} = {x_i}")

    k = network.next_round()

network.stop()
```

Other supported algorithms include:

* `DGD` (Distributed Gradient Descent) [[2]](#references).
* `NIDS` (Network Independent Step-size) [[3]](#references).
* `DIGing` (Distributed Inexact Gradient method with gradient tracking) [[4]](#references).
* `AugDGM` (Augmented Distributed Gradient Method) [[5]](#references).
* `WE` (Wang–Elia) [[6]](#references).
* `RGT` (Robust Gradient Tracking) [[7]](#references).
* `RAugDGM`, a robust variant of `AugDGM` and the `ATC` (Adapt-Then-Combine) variant of `RGT`.
* `AtcWE`, the `ATC` variant of `WE`.

### Optional: Construct `neighbors` from the Mixing Matrix

If the network topology is given as a mixing matrix $W$, the `Graph` class can be used as a helper to convert it into neighbor information.

For example, `graph[i]` returns the neighbors and corresponding weights for node `i`.

```python
import numpy as np

from conops import Graph


L = np.array([
    [ 2, -1,  0, -1],
    [-1,  2, -1,  0],
    [ 0, -1,  2, -1],
    [-1,  0, -1,  2],
])

W = np.eye(4) - 0.2 * L

graph = Graph.from_mixing_matrix(W)

# Neighbors of node 0
neighbors = graph[0]

print(neighbors)
```

## Distributed Execution and Communication

DisCoOpt relies on [ConOps](https://github.com/rui-huang-opt/conops) for synchronous neighbor-to-neighbor communication.

ConOps provides the communication layer used by the distributed optimization algorithms, allowing each node to exchange information directly with its neighbors according to the specified network topology.

The same DisCoOpt algorithms can be executed either across multiple machines or as multiple processes on a single machine.

For local experiments, the example notebooks use [Ray](https://www.ray.io/) to launch and manage multiple node processes on one machine.

Ray can also be used to orchestrate nodes across multiple machines, while the actual neighbor-to-neighbor communication and network operations are handled by ConOps.

For implementation details and example configurations, please refer to the sample notebooks in the [`examples`](./examples/) directory.

## License

This project is licensed under the MIT License.

## References

[1] W. Shi, Q. Ling, G. Wu, and W. Yin, "EXTRA: An exact first-order algorithm for decentralized consensus optimization," *SIAM Journal on Optimization*, vol. 25, no. 2, pp. 944–966, 2015.
[Link](https://epubs.siam.org/doi/abs/10.1137/14096668X)

[2] A. Nedic and A. Ozdaglar, "Distributed subgradient methods for multi-agent optimization," *IEEE Transactions on Automatic Control*, vol. 54, no. 1, pp. 48–61, 2009.
[Link](https://ieeexplore.ieee.org/document/4749425)

[3] Z. Li, W. Shi, and M. Yan, "A decentralized proximal-gradient method with network independent step-sizes and separated convergence rates," *IEEE Transactions on Signal Processing*, vol. 67, no. 17, pp. 4494–4506, 2019.
[Link](https://ieeexplore.ieee.org/document/8752033)

[4] A. Nedic, A. Olshevsky, and W. Shi, "Achieving geometric convergence for distributed optimization over time-varying graphs," *SIAM Journal on Optimization*, vol. 27, no. 4, pp. 2597–2633, 2017.
[Link](https://epubs.siam.org/doi/abs/10.1137/16M1084316)

[5] J. Xu, S. Zhu, Y. C. Soh, and L. Xie, "Augmented distributed gradient methods for multi-agent optimization under uncoordinated constant stepsizes," in *Proceedings of the 54th IEEE Conference on Decision and Control (CDC)*, pp. 2055–2060, 2015.
[Link](https://ieeexplore.ieee.org/document/7402509)

[6] J. Wang and N. Elia, "Control approach to distributed optimization," in *Proceedings of the 48th Annual Allerton Conference on Communication, Control, and Computing*, pp. 557–561, 2010.
[Link](https://ieeexplore.ieee.org/document/5706956)

[7] S. Pu, "A robust gradient tracking method for distributed optimization over directed networks," in *Proceedings of the 59th IEEE Conference on Decision and Control (CDC)*, pp. 2335–2341, 2020.
[Link](https://ieeexplore.ieee.org/document/9303917)
