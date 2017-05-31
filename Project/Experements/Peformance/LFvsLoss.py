import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import random

import sys
sys.path.append('../lib/')

import MatrixFactorization as MatrixDecomposition

plt.switch_backend("TkAgg")

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0

##R[2,3] = 0
##R[4,7] = 0
##R[5,2] = 0
##R[9,4] = 0
##R[3,2] = 0
##R[1,1] = 0
##R[9,6] = 0
#l, dcomp = MatrixDecomposition.factorize_matrix(R, 4)
#print(l)

#print(R)
#print(np.dot(dcomp[0],dcomp[1]))

zero_out(R, 80)
print(R)

LF_start = 1
LF_end = 8
repeat = 10

avg_losses = []
std_losses = []
for i in range(LF_start, LF_end):
    print(i)
    loss = []
    for j in range(repeat):
        l, dcomp = MatrixDecomposition.factorize_matrix(R, i)
        loss.append(l)

    avg_losses.append(np.array(loss).mean())
    std_losses.append(np.array(loss).std())


x_axis = np.array(range(LF_start, LF_end))
df = np.repeat(repeat-1, LF_end - LF_start)

plt.plot(x_axis, avg_losses, '-o', color='b', )
plt.errorbar(x_axis, avg_losses, yerr=ss.t.ppf(0.95, df)*std_losses)
plt.show()
