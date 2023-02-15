import numpy as np


class NN:
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2
        self.z = []
        self.a = []
        self.deltas = []

    def forward(self, inputs):
        self.a.append(inputs)
        z_0 = self.w1.dot(inputs)
        a_0 = [max(z_0[0], 0), self.sigmoid(z_0[1])]
        z_1 = self.w2.dot(a_0)
        pred = self.sigmoid(z_1)

        self.z.append(z_0)
        self.a.append(a_0)
        self.z.append(z_1)
        self.a.append(pred)

        return pred

    def backward(self, pred, y):
        delta2 = self.grad_J_a(pred, y) * self.sigmoid_prime(self.z[1])
        delta1 = self.w2.T.dot(
            delta2) * np.array([self.max_prime(self.z[0][0]), self.sigmoid_prime(self.z[0][1])])
        self.deltas.append(delta1)
        self.deltas.append(delta2)

    def w_der(self, l, j, k):
        return self.a[l-2][k-1]*self.deltas[l-2][j-1]

    def sigmoid(self, x):
        """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def max_prime(self, x):
        if x > 0:
            return 1
        return 0

    def grad_J_a(self, pred, y):
        return pred - y


w1 = np.array([[0.7, 0.2, 0.7], [0.8, 0.3, 0.6]])
w2 = np.array([0.2, 0.4])
neural_network = NN(w1, w2)
pred = neural_network.forward(np.array([0, 1, 1]))
neural_network.backward(pred, 1)
print(neural_network.w_der(2, 1, 3))
