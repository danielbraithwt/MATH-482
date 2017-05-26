
import sys
sys.path.append('../lib/')

import MatrixFactorization as MatrixDecomposition
import MatrixFactorizationSNN
import MatrixFactorizationMNN
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))


repeat = 0

#dcomp_m = []
#snn_m = []
#mnn_m = []

dcomp_losses = np.array(MatrixDecomposition.factorize_matrix(R, 4))
snn_losses = np.array(MatrixFactorizationSNN.factorize_matrix(R, 8))
mnn_losses = np.array(MatrixFactorizationMNN.factorize_matrix(R, 4))

#for i in range(1, repeat):
#    print(i)
#    dcomp_losses = np.add(dcomp_losses, MatrixDecomposition.factorize_matrix(R, 4))
#    snn_losses = np.add(snn_losses, MatrixFactorizationSNN.factorize_matrix(R, 8))
#    mnn_losses = np.add(mnn_losses, MatrixFactorizationMNN.factorize_matrix(R, 4))

#dcomp_losses = dcomp_losses/float(R)
#snn_losses = snn_losses/float(R)
#mnn_losses = mnn_losses/float(R)

x_axis = np.array(range(0, 10001, 100))

snn_losses = np.delete(snn_losses, 0)

print(dcomp_losses)
print(snn_losses)
print(mnn_losses)

plt.plot(x_axis, dcomp_losses, label="Decomposition")
plt.plot(x_axis, snn_losses, label="Single Neural Network")
plt.plot(x_axis, mnn_losses, label="Multiple Neural Networks")
plt.legend()
plt.show()
