import jax.numpy as jnp
from jax import random

def init_mlp_params(key, sizes):
    keys = random.split(key, len(sizes) - 1)
    return [(random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jnp.maximum(x, 0)  # ReLU
    w_last, b_last = params[-1]
    return jnp.dot(x, w_last) + b_last
