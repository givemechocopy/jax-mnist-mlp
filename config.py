import jax

config = {
    "key": jax.random.PRNGKey(42),
    "model_arch": [784, 256, 128, 10],
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.01
}
