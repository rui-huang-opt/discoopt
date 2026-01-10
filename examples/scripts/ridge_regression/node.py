import numpy as np
from numpy.typing import NDArray

# Create a simple graph
node_names = ["1", "2", "3", "4"]

# Set parameters for ridge regression
dim = 10

np.random.seed(0)

rho = 0.01
u = {i: np.random.uniform(-1, 1, dim) for i in node_names}
x_tilde = {i: np.multiply(0.1 * (int(i) - 1), np.ones(dim)) for i in node_names}
epsilon = {i: np.random.normal(0, 5) for i in node_names}
v = {i: u[i] @ x_tilde[i] + epsilon[i] for i in node_names}

# Obtain command line arguments
import argparse


class Args(argparse.Namespace):
    node_id: str
    gamma: float
    max_iter: int


parser = argparse.ArgumentParser()
parser.add_argument("node_id", type=str, help="Name of the node")
parser.add_argument("--gamma", type=float, default=0.31, help="Step size")
parser.add_argument("--max_iter", type=int, default=2000, help="Maximum iterations")
args = parser.parse_args(namespace=Args())

# Configure logging to display INFO level messages
import logging

logging.basicConfig(level=logging.INFO)


# Distributed optimization
def f(var: NDArray[np.float64]) -> NDArray[np.float64]:
    return (u[args.node_id] @ var - v[args.node_id]) ** 2 + rho * var @ var


from topolink import NodeHandle

nh = NodeHandle(args.node_id)

from dco import AugDGM

optimizer = AugDGM(f, nh, args.gamma)

x_i = np.zeros(dim)

optimizer.init(x_i)

for k in range(args.max_iter):
    x_i = optimizer.step(x_i)
    print(f"Node {args.node_id} iteration {k}: x = {x_i}")
