import numpy as np
from torchvision import datasets

def get_mnist_data():
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    X = np.array(mnist.data.numpy(), dtype=np.float32) / 255.0
    y = np.array(mnist.targets.numpy())
    X = X.reshape(-1, 784)

    return X[:50000], y[:50000], X[50000:], y[50000:]


def get_mnist_test_data():
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    X_test = np.array(mnist_test.data.numpy(), dtype=np.float32) / 255.0
    y_test = np.array(mnist_test.targets.numpy())
    X_test = X_test.reshape(-1, 784)
    return X_test, y_test
