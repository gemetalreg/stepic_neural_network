import numpy as np
from numpy import linalg
from urllib.request import urlopen

link = input()
f = urlopen(link)
data = np.loadtxt(f, skiprows=1, delimiter=",")

Y = data[:,0]
X = np.hstack((np.ones((data.shape[0],1)), data[:,1:]))

b = linalg.inv(X.T@X)@X.T@Y

print(*b)