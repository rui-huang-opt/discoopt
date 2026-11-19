# Distributed Composite Optimization (DisCoOpt)

DisCoOpt is a Python package for distributed composite optimization, designed to solve problems of the form

$$
\min_{x \in \mathbb{R}^d}
\quad
\sum_{i=1}^n \left(f_i(x) + g_i(x)\right),
$$

where each $f_i:\mathbb{R}^d\to\mathbb{R}$ is a smooth local objective function, and each $g_i:\mathbb{R}^d\to(-\infty,+\infty]$ is a possibly nonsmooth regularization term.

DisCoOpt provides implementations of classical and robust distributed optimization algorithms, supporting research in decentralized learning, multi-agent systems, and networked optimization.

The package also includes implementations used in the numerical experiments of the following papers:

- **Robust Distributed Composite Optimization: A Unified Framework with Linear Convergence and Stability Analysis.**
  *IEEE Transactions on Automatic Control*, accepted for publication.
- [**A Unified Framework for Robust Distributed Optimization under Bounded Disturbances**](https://ieeexplore.ieee.org/abstract/document/11178647).

## Features

- Classical and robust distributed optimization algorithms.
- Support for smooth and composite optimization problems.
- Distributed execution across multiple machines or local processes.
- A modular and extensible architecture for developing new algorithms.
- A formula-style API built on NumPy and JAX, allowing distributed optimization algorithms to be implemented in a way that closely follows their mathematical expressions.
- Decentralized neighbor-to-neighbor communication through [ConOps](https://github.com/rui-huang-opt/conops).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/rui-huang-opt/discoopt.git
```

For development:

```bash
git clone https://github.com/rui-huang-opt/discoopt.git
cd discoopt
pip install -e .
```

## Usage Example

The following example demonstrates distributed ridge regression over a network of four agents:

$$
\min_{x\in\mathbb{R}^d}
\quad
\sum_{i=1}^4
\left(
(u_i^\top x-v_i)^2+\rho\|x\|^2
\right).
$$

Each agent maintains a local copy $x_i$ and evaluates its own objective

$$
f_i(x_i)=(u_i^\top x_i-v_i)^2+\rho\|x_i\|^2.
$$

We use the EXTRA algorithm [1], whose iteration can be written as

$$
\mathbf{x}^{k+2}
=
(I+W)\mathbf{x}^{k+1}
-
\frac{I+W}{2}\mathbf{x}^{k}
-
\gamma\left(
\nabla F(\mathbf{x}^{k+1})
-
\nabla F(\mathbf{x}^{k})
\right),
$$

where $W$ is a symmetric mixing matrix determined by the communication topology, and $F$ denotes the stacked local objective.

Each node runs the same program with its own local data, node identifier, and neighbor configuration.

```python
import numpy as np
from numpy.typing import NDArray

from conops import Network
from dco import EXTRA

# Node-specific data
node_id = "1"

u_i = np.array([1.0, 2.0, 3.0])
v_i = np.array([1.0])

rho = 0.1
dimension = 3
step_size = 0.01

# Neighbor weights
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

try:
    k = network.round_id

    while k < 500:
        x_i = optimizer.step(x_i)

        print(f"Step {k}, x_{node_id} = {x_i}")

        k = network.next_round()
finally:
    network.stop()
```

### Supported Algorithms

DisCoOpt includes implementations of the following distributed optimization algorithms:

| Algorithm | Description | Reference |
|---|---|---|
| `DGD` | Distributed Gradient Descent | [2] |
| `EXTRA` | Exact First-Order Algorithm | [1] |
| `NIDS` | Network Independent Step-size | [3] |
| `DIGing` | Distributed Gradient Tracking | [4] |
| `AugDGM` | Augmented Distributed Gradient Method | [5] |
| `WE` | Wang–Elia Algorithm | [6] |
| `RGT` | Robust Gradient Tracking | [7] |
| `RAugDGM` | Robust AugDGM; ATC counterpart of RGT | RDCO |
| `AtcWE` | Adapt-Then-Combine variant of WE | RDCO |

The robust algorithms are closely related to the **Robust Distributed Composite Optimization (RDCO)** framework, which provides a unified interpretation of their robustness properties and supports extensions to composite optimization.

### Optional: Construct Neighbors from a Mixing Matrix

When the network topology is specified by a mixing matrix $W$, the `Graph` class from ConOps can be used to construct neighbor configurations.

For example:

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

DisCoOpt uses [ConOps](https://github.com/rui-huang-opt/conops) as its communication backend.

ConOps handles synchronous neighbor-to-neighbor communication, allowing optimization algorithms to exchange local states according to the specified network topology.

This separation between optimization and communication enables the same algorithm implementations to run in different execution environments:

- **Single-machine execution:** Multiple agents run as independent local processes.
- **Multi-machine execution:** Agents run on different computers and communicate over a network.

The example notebooks use [Ray](https://www.ray.io/) to launch and manage multiple agents. Ray can also be used for multi-machine orchestration, while ConOps handles the actual neighbor-to-neighbor communication.

For implementation details and example configurations, see the [`examples`](./examples/) directory.

## License

This project is licensed under the MIT License.

## References

[1] W. Shi, Q. Ling, G. Wu, and W. Yin, "EXTRA: An exact first-order algorithm for decentralized consensus optimization," *SIAM Journal on Optimization*, vol. 25, no. 2, pp. 944–966, 2015. [Link](https://epubs.siam.org/doi/abs/10.1137/14096668X)

[2] A. Nedic and A. Ozdaglar, "Distributed subgradient methods for multi-agent optimization," *IEEE Transactions on Automatic Control*, vol. 54, no. 1, pp. 48–61, 2009. [Link](https://ieeexplore.ieee.org/document/4749425)

[3] Z. Li, W. Shi, and M. Yan, "A decentralized proximal-gradient method with network independent step-sizes and separated convergence rates," *IEEE Transactions on Signal Processing*, vol. 67, no. 17, pp. 4494–4506, 2019. [Link](https://ieeexplore.ieee.org/document/8752033)

[4] A. Nedic, A. Olshevsky, and W. Shi, "Achieving geometric convergence for distributed optimization over time-varying graphs," *SIAM Journal on Optimization*, vol. 27, no. 4, pp. 2597–2633, 2017. [Link](https://epubs.siam.org/doi/abs/10.1137/16M1084316)

[5] J. Xu, S. Zhu, Y. C. Soh, and L. Xie, "Augmented distributed gradient methods for multi-agent optimization under uncoordinated constant stepsizes," in *Proceedings of the 54th IEEE Conference on Decision and Control (CDC)*, pp. 2055–2060, 2015. [Link](https://ieeexplore.ieee.org/document/7402509)

[6] J. Wang and N. Elia, "Control approach to distributed optimization," in *Proceedings of the 48th Annual Allerton Conference on Communication, Control, and Computing*, pp. 557–561, 2010. [Link](https://ieeexplore.ieee.org/document/5706956)

[7] S. Pu, "A robust gradient tracking method for distributed optimization over directed networks," in *Proceedings of the 59th IEEE Conference on Decision and Control (CDC)*, pp. 2335–2341, 2020. [Link](https://ieeexplore.ieee.org/document/9303917)