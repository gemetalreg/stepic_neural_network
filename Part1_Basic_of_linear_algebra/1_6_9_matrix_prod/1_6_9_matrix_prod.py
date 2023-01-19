import numpy as np
import sys

def matrix_prod():
    stdin = sys.stdin
    sys.stdin = open("test2.txt", "rt")

    x_shape = tuple(map(int, input().split()))
    X = np.fromiter(map(int, input().split()), np.int16).reshape(x_shape)

    y_shape = tuple(map(int, input().split()))
    Y = np.fromiter(map(int, input().split()), np.int16).reshape(y_shape)
    Y_t = Y.T

    if X.shape[1] != Y_t.shape[0]:
        print("matrix shapes do not match")
    else:
        print(X@Y_t)

    sys.stdin = stdin

matrix_prod() 