import jax.numpy as jnp
import jax

def cross_entropy(params, x, y, model_fn):
    logits = model_fn(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))

def accuracy_fn(params, x, y, model_fn):
    pred = jnp.argmax(model_fn(params, x), axis=1)
    return jnp.mean(pred == y)
