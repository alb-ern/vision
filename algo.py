import numpy as np


def normalize(i):
    return i / 127.5 - 1


def rrelu(x):
    # assuming x is float or int numpy array
    negative_slope = 0.01
    return np.where(x > 0, x, x * negative_slope)


def softmax_rows(arr):
    # subtract max per row for numerical stability
    exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


class Agent:
    def __init__(self) -> None:
        self.error = 0
        self.w1 = np.random.random((784, 784))
        self.b1 = np.random.random(784)
        self.w2 = np.random.random((784, 196))
        self.b2 = np.random.random(196)
        self.w3 = np.random.random((196, 10))
        self.b3 = np.random.random(10)

    def forward(self, batch):
        images, labels = batch
        images = images.reshape(len(images), -1) / 127.5 - 1
        layer1 = rrelu((images.dot(self.w1) + self.b1) / 785)
        layer2 = rrelu((layer1.dot(self.w2) + self.b2) / 785)
        out = softmax_rows((layer2.dot(self.w3) + self.b3) / 197)
        self.error = sum((1 - out[labels, np.arange(10)])**2)
        return out

    def backward(self):
        return
