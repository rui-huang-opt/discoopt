# Distributed Composite Optimization (DisCoOpt)
Distributed Composite Optimization (DisCoOpt) is a Python package for solving composite optimization problems of the form

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{n} \sum_{i=1}^n f_i(x) + g(x),
$$

where each $f_i: \mathbb{R}^d \rightarrow \mathbb{R}$ is a smooth loss function associated with a local dataset or agent, and $g(x)$ is a (possibly non-smooth) regularization term.
DisCoOpt enables efficient and robust distributed optimization across multiple nodes, making it suitable for federated learning, multi-agent systems, and large-scale machine learning tasks.

This package contains the experimental code for the paper [*A Unified Framework for Robust Distributed Optimization under Bounded Disturbances*](https://ieeexplore.ieee.org/abstract/document/11178647).

## Features

- Support for distributed and parallel optimization
- Modular and extensible architecture
- Formula-style API based on NumPy and JAX, allowing you to define and deploy distributed optimization algorithms across multiple machines as naturally as writing mathematical expressions

## Installation
Install via pip:

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

This example demonstrates distributed ridge regression, where each node solves a local least squares problem with $\ell_2$ regularization:

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{4} \sum_{i = 1}^4 (u_i^\top x - v_i)^2 + \rho \| x \|^2.
$$

We use the `EXTRA` (**EX**act firs**T**-orde**R** **A**lgorithm) [[1]](#references) to solve this problem:

$$
\mathbf{x}^{k + 2} = (I + W) \mathbf{x}^{k + 1} - \frac{I + W}{2} \mathbf{x}^k - \gamma \big(\nabla f(\mathbf{x}^{k + 1}) - \nabla f(\mathbf{x}^{k})\big),
$$

where $W$ is a symmetric mixing matrix determined by the network topology.

Each node defines its own local data and optimization parameters, including the feature vector `u_i`, target value `v_i`, regularization parameter `rho`, and step size.
The local objective function at node $i$ is

$$
f_i(x_i) = (u_i^\top x_i - v_i)^2 + \rho \|x_i\|^2.
$$

In the following example, we use the `EXTRA` [[1]](#references) algorithm.
Other supported algorithms include:

- `DGD` (**D**istributed **G**radient **D**escent) [[2]](#references)
- `NIDS` (**N**etwork **I**n**D**ependent **S**tep-size) [[3]](#references)
- `DIGing` (**D**istributed **I**nexact **G**radient method with gradient track**ing**) [[4]](#references)
- `AugDGM` (**Aug**mented **D**istributed **G**radient **M**ethod) [[5]](#references)
- `WE` (**W**ang-**E**lia) [[6]](#references)
- `RGT` (**R**obust **G**radient **T**racking) [[7]](#references)
- `RAugDGM`: a robust variant of `AugDGM` and the `ATC` (**A**dapt-**T**hen-**C**ombine) variant of `RGT`
- `AtcWE`: the `ATC` variant of `WE`

Algorithms are selected explicitly by importing the corresponding optimizer class. For example:

```python
import numpy as np
from numpy.typing import NDArray
from conops import NodeHandle
from discoopt import EXTRA

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

nh = NodeHandle(node_id, neighbors)
optimizer = EXTRA(f_i, nh, step_size)

x_i = np.zeros(dimension)
optimizer.init(x_i)

for k in range(500):
    x_i = optimizer.step(x_i)
    print(f"Step {k}, x_{node_id} = {x_i}")
```

### Optional: construct `neighbors` from the mixing matrix $W$

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

# neighbors of node 0
neighbors = graph[0]
print(neighbors)
```

### Running Distributed Algorithms with Ray

If you do not have access to multiple machines, you can still run distributed optimization algorithms on a single machine using multiple processes.
The notebook examples also support multi-machine execution with Ray.

In either case, the key configuration is the `transport` parameter of `NodeHandle`:

- use `"ipc"` for communication between processes on a single machine
- use `"tcp"` for communication across machines over the network

For implementation details and example configurations, please refer to the sample notebooks in the [`examples/notebooks`](./examples/notebooks/) directory.

## License

This project is licensed under the MIT License.

## References

[1] [Shi, W., Ling, Q., Wu, G., & Yin, W. (2015). Extra: An exact first-order algorithm for decentralized consensus optimization. SIAM Journal on Optimization, 25(2), 944-966.](https://epubs.siam.org/doi/abs/10.1137/14096668X)

[2] [Nedic, A., & Ozdaglar, A. (2009). Distributed subgradient methods for multi-agent optimization. IEEE Transactions on automatic control, 54(1), 48-61.](https://ieeexplore.ieee.org/abstract/document/4749425?casa_token=epANmmnAinUAAAAA:FX3PsEIp9OslefhUYQvZUqCWiOq71NulcjsRd3FmplTrI5eq1U9_R0sGm6mYaKnZkLBk6Jmvu8TB)

[3] [Li, Z., Shi, W., & Yan, M. (2019). A decentralized proximal-gradient method with network independent step-sizes and separated convergence rates. IEEE Transactions on Signal Processing, 67(17), 4494-4506.](https://ieeexplore.ieee.org/abstract/document/8752033?casa_token=wuimAh5R6q4AAAAA:3H5eQjethbj18lYd7gFwxWJfpbIY9wrbnpdK0ngv4d9-pKAYnptBEOxd9t41SvwlG954NxsrR43b)

[4] [Nedic, A., Olshevsky, A., & Shi, W. (2017). Achieving geometric convergence for distributed optimization over time-varying graphs. SIAM Journal on Optimization, 27(4), 2597-2633.](https://epubs.siam.org/doi/abs/10.1137/16M1084316)

[5] [Xu, J., Zhu, S., Soh, Y. C., & Xie, L. (2015, December). Augmented distributed gradient methods for multi-agent optimization under uncoordinated constant stepsizes. In 2015 54th IEEE Conference on Decision and Control (CDC) (pp. 2055-2060). IEEE.](https://ieeexplore.ieee.org/abstract/document/7402509?casa_token=ZyrMzbDNhZAAAAAA:vUhyL-hMDY_rQdXk243Yqa4vR1LXX6SRa_kA1-P9uLuhIiYBj8GtLcXwQpSSYBnms9EQnBhWFz8o)

[6] [Wang, J., & Elia, N. (2010, September). Control approach to distributed optimization. In 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton) (pp. 557-561). IEEE.](https://ieeexplore.ieee.org/abstract/document/5706956?casa_token=vvUwmPdgUZYAAAAA:4SkBjSGeMCioniqMTx9wj9I6eWCsQ3FyfZ08uFv6z3t0VVXEGVNBHh_Oz7shF2EdmtBy5psYV6EP)

[7] [Pu, S. (2020, December). A robust gradient tracking method for distributed optimization over directed networks. In 2020 59th IEEE Conference on Decision and Control (CDC) (pp. 2335-2341). IEEE.](https://ieeexplore.ieee.org/abstract/document/9303917?casa_token=BCK5Zf8Tgx4AAAAA:DNJiCu2VQehdxeELc2i2enKSpKEE8SBcSIPIHlrfz4GFLE8DqjbYycqh3PNaY5NopFR9WyZV5G05)