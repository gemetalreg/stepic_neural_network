import numpy as np
from urllib.request import urlopen

f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
data = np.loadtxt(f, skiprows=1, delimiter=",")

print(data.mean(axis=0))