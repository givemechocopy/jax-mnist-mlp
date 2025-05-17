import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad
from utils.metrics import cross_entropy, accuracy_fn

def train(params, X_train, y_train, X_val, y_val, X_test, y_test, config):
    optimizer = config['optimizer']
    opt_state = optimizer.init(params)

    def loss_fn(p, x, y):
        return cross_entropy(p, x, y, config['model'])

    @jax.jit
    def update(params, opt_state, x, y):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss_history = []
    val_acc_history = []
    test_acc_history = []

    for epoch in range(config['epochs']):
        for i in range(0, len(X_train), config['batch_size']):
            x_batch = jnp.array(X_train[i:i + config['batch_size']])
            y_batch = jnp.array(y_train[i:i + config['batch_size']])
            params, opt_state, loss = update(params, opt_state, x_batch, y_batch)

        val_acc = accuracy_fn(params, jnp.array(X_val), jnp.array(y_val), config['model'])
        val_loss = cross_entropy(params, jnp.array(X_val), jnp.array(y_val), config['model'])
        test_acc = accuracy_fn(params, jnp.array(X_test), jnp.array(y_test), config['model'])

        val_acc_history.append(float(val_acc))
        test_acc_history.append(float(test_acc))
        loss_history.append(float(val_loss))

        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    return params, loss_history, val_acc_history, test_acc_history
