# Distributed Composite Optimization (DisCoOpt)
Distributed Composite Optimization (DCO) is a Python package for solving composite optimization problems of the form

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{n} \sum_{i=1}^n f_i(x) + g(x),
$$

where each $f_i: \mathbb{R}^d \rightarrow \mathbb{R}$ is a smooth loss function associated with a local dataset or agent, and $g(x)$ is a (possibly non-smooth) regularization term. DCO enables efficient and robust distributed optimization across multiple nodes, making it suitable for federated learning, multi-agent systems, and large-scale machine learning tasks.

This package contains the experimental code for the paper *A Unified Framework for Robust Distributed Optimization under Bounded Disturbances*.

## Features

- Support for distributed and parallel optimization
- Modular and extensible architecture
- Formula-style API based on NumPy and autograd (JAX), allowing you to define and deploy distributed optimization algorithms across multiple machines as naturally as writing mathematical expressions

## Installation
Install via pip:

```bash
pip install git+https://github.com/rui-huang-opt/discoopt.git
```

Or, for development:

```bash
git clone https://github.com/rui-huang-opt/discoopt.git
cd dco
pip install -e .
```

## Usage Example

The typical workflow of DCO consists of two main steps:

1. **Defining the network topology**: Specify the communication structure among nodes in the distributed system.
2. **Defining the local problem at each node**: Set up the local objective function and related parameters for each node individually.

The following example demonstrates distributed ridge regression, where each node solves a local least squares problem with $\ell_2$ regularization:

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{4} \sum_{i = 1}^4 (u_i^\top x - v_i)^2 + \rho \| x \|^2.
$$

The distributed ridge regression problem can be efficiently addressed using the `EXTRA` (**EX**act firs**T**-orde**R** **A**lgorithm) [[1]](#references):

$$
\mathbf{x}^{k + 2} = (I + W) \mathbf{x}^{k + 1} - \frac{I + W}{2} \mathbf{x}^k - \gamma [\nabla f(\mathbf{x}^{k + 1}) - \nabla f(\mathbf{x}^{k})],
$$

where $W$ is a symmetric mixing matrix determined by the network topology.

Below are code templates for both steps

### 1. Specify the network topology (mixing matrix $W$) on the server

The server only assists nodes in establishing connections to form an undirected graph network. It does not participate in communication during computation.
For more details, please refer to the [`topolink`](https://github.com/rui-huang-opt/topolink) repository.

```python
import numpy as np
from logging import basicConfig, INFO
from topolink import Graph

basicConfig(level=INFO)

L = np.array([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
W = np.eye(4) - L * 0.2

graph = Graph.from_mixing_matrix(W)

graph.deploy()

```

### 2. Define the local optimization problem at each node
Each node specifies its own data and parameters, such as the local feature vector `u_i`, target value `v_i`, regularization parameter `rho`, and the optimization step size.
The local objective function `f_i(x_i)` is defined using these parameters as

$$
f_i(x_i) = (u_i^\top x_i - v_i)^2 + \rho \| x_i \|^2.
$$

The `LocalObjective` class allows you to specify the smooth part of the objective, and you can set the `g_type` parameter to include different types of regularization, such as `"zero"` (no regularization, default value), `"l1"` (L1 regularization), or others as needed.
The `Optimizer` class is then used to set up and solve the optimization problem.

We use the `EXTRA` [[1]](#references) algorithm as an example here.
The package also implements several other distributed optimization algorithms, including:

- `DGD` (**D**istributed **G**radient **D**escent) [[2]](#references)
- `NIDS` (**N**etwork **I**n**D**ependent **S**tep-size) [[3]](#references)
- `DIGing` (**D**istributed **I**nexact **G**radient method and a gradient track**ing**) [[4]](#references)
- `AugDGM` (**Aug**mented **D**istributed **G**radient **M**ethods) [[5]](#references)
- `WE` (**W**ang-**E**lia) [[6]](#references)
- `RGT` (**R**obust **G**radient **T**racking) [[7]](#references)

In addition, the package includes two original algorithms proposed in our paper:

- `RAugDGM`: A robust version of `AugDGM` algorithm and the `ATC` (**A**dapt-**T**hen-**C**ombine) versionl of `RGT` algorithm.
- `AtcWE`: The `ATC` version of `WE` algorithm.

You can select the algorithm by setting the `algorithm` parameter in the `Optimizer.create` method. If you do not specify the `algorithm` parameter, the default algorithm used is `"RAugDGM"`.
Please fill in the details of the additional algorithms as needed.

```python
# Distributed optimization example

from logging import basicConfig, INFO
basicConfig(level=INFO)

import numpy as np
from numpy.typing import NDArray
from discoopt import LossFunction, Optimizer
from topolink import NodeHandle

# Node-specific data (replace with your own)
node_id = "1"
u_i = np.array([1.0, 2.0, 3.0])
v_i = np.array([1.0])
rho = 0.1
dimension = 3
step_size = 0.01

# ===============================
# Approach 1: Add regularization directly in the objective
# ===============================
def f_i_direct(x_i: NDArray[np.float64]) -> NDArray[np.float64]:
    return (u_i @ x_i - v_i) ** 2 + rho * x_i @ x_i

loss_fn_direct = LossFunction(f_i_direct)

# ===============================
# Approach 2: Use the L2 regularization parameter in LossFunction
# ===============================
def f_i_l2(x_i: NDArray[np.float64]) -> NDArray[np.float64]:
    return (u_i @ x_i - v_i) ** 2

loss_fn_l2 = LossFunction(f_i_l2, g_type="l2", lam=rho)

# Both approaches are equivalent; you can use either `loss_fn_direct` or `loss_fn_l2`
loss_fn = loss_fn_l2  # choose one to run

# Network handle
nh = NodeHandle(node_id)

# Optimizer setup
optimizer = Optimizer.create(loss_fn, nh, step_size, algorithm="EXTRA")

x_i = np.zeros(dimension)

# Initialize optimizer (sets up internal variables using x_i)
optimizer.init(x_i)

# Optimization loop
for k in range(500):
    x_i = optimizer.step(x_i)
    print(f"Step {k}, x_{node_id} = {x_i}")
```

### Running Distributed Algorithms with Multiple Processes on a Single Machine

If you do not have access to multiple machines, you can still experiment with and test distributed optimization algorithms by launching multiple processes on a single machine. Each process acts as an independent node and communicates with others via network ports.
For implementation details and configuration examples, please refer to the sample code in the [`examples/notebooks`](./examples/notebooks/) directory.

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