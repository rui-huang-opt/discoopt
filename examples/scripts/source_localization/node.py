import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

# Set parameters for the source localization problem
A = 100
src_loc = np.array([10, 40])

# Sensor locations
n_sens = 10

np.random.seed(0)
sens_loc_x = np.random.uniform(-10, 30, n_sens)
sens_loc_y = np.random.uniform(20, 60, n_sens)
sens_loc = np.vstack((sens_loc_x, sens_loc_y))

# Generate the measurements
n_meas = 15
meas_var = 1
meas_normal = A / norm(src_loc[:, np.newaxis] - sens_loc, axis=0)
meas = meas_normal[np.newaxis, :] + np.random.normal(0, meas_var, (n_meas, n_sens))

# Regularization parameter
rho = 0.0001

# Obtain node name from command line arguments
import sys

if len(sys.argv) > 1:
    node_id = "".join(sys.argv[1:])
else:
    print("Usage: python node.py <node_name>")
    sys.exit(1)

# Configure logging to display INFO level messages
from logging import basicConfig, INFO

basicConfig(level=INFO)

# Distributed optimization
import jax.numpy as jnp
from jax import Array

gamma = 0.085
max_iter = 7000
meas_i = meas[:, int(node_id) - 1]
sens_loc_i = sens_loc[:, int(node_id) - 1]


def f(var: NDArray[np.float64]) -> Array:
    signal_diff = meas_i - A / jnp.linalg.norm(var - sens_loc_i)
    regularizer = rho * jnp.sum(jnp.square(var))

    return jnp.mean(signal_diff**2) + regularizer


from topolink import NodeHandle

nh = NodeHandle(name=node_id)

from discoopt.optimizer import RAugDGM

optimizer = RAugDGM(f, nh, gamma)

theta_i = np.zeros(2)

optimizer.init(theta_i)
for k in range(max_iter):
    theta_i = optimizer.step(theta_i)
    print(f"Node {node_id} iteration {k}: theta = {theta_i}")
