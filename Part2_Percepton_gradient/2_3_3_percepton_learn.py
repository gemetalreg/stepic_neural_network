import numpy as np

def summation_func(w, x):
    return w.T.dot(x)

def activate_func(res):
    if res > 0:
        return 1
    else:
        return 0

data = np.array([
    [1, 0.3, 1],
    [0.4, 0.5, 1],
    [0.7, 0.8, 0]
])

w = np.array([0,0,0], dtype=float)
y = data[:,2]
X = data[:,:2]
one = np.ones((data.shape[0],1))
X = np.hstack((one, X))

for i in range(X.shape[0]):
    s = summation_func(w, X[i])
    a = activate_func(s)
    if  a != y[i]:
        if a == 0:
            w += X[i]
        if a == 1:
            w -= X[i]

print(w)
