import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import random

import sys
sys.path.append('../lib/')

import MatrixFactorizationMNN

plt.switch_backend("TkAgg")

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0

#print(MatrixFactorizationMNN.factorize_matrix(R, 5, 5))

#zero_out(R, 80)
#print(R)

lf_start = 1
lf_end = 11
repeat = 10

avg_losses = []
std_losses = []
for i in range(lf_start, lf_end):
    print(i)
    loss = []
    for j in range(repeat):
        print(j)
        l = MatrixFactorizationMNN.factorize_matrix(R, i, 7)
        print(l)
        loss.append(l)

    print()
    avg_losses.append(np.array(loss).mean())
    std_losses.append(np.array(loss).std())


x_axis = np.array(range(lf_start,lf_end))
df = np.repeat(repeat-1, lf_end - lf_start)

plt.plot(x_axis, avg_losses, '-o', color='b', )
plt.errorbar(x_axis, avg_losses, yerr=ss.t.ppf(0.95, df)*std_losses)
plt.show()
