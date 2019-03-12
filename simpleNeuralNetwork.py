import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
    [0,0,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
    [0]
])

np.random.seed(1)       # 指定随机种子
w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

for i in range(60001):
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, w0))
    layer_2 = sigmoid(np.dot(layer_1, w1))

    error = y - layer_2
    if i % 10000 == 0:
        print("after %d step(s) training the error is + %s" % (i, str(np.mean(np.abs(error)))))

    layer_2_delta = error * sigmoid(layer_2, True)
    l1_error = layer_2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(layer_1, True)

    w1 += layer_1.T.dot(layer_2_delta)
    w0 += layer_0.T.dot(l1_delta)
