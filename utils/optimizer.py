import optax

def create_optimizer(learning_rate: float):
    return optax.adam(learning_rate)