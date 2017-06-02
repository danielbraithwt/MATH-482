import sys
import numpy as np
sys.path.append('../lib/')

import MatrixFactorizationMNN
import MatrixFactorization as MatrixDecomposition
import MatrixFactorizationSNN

import MovieLensData as RealData

def run_tests(R, lf, repeats):
    dcomp_peformance = []
    single_net_peformance = []
    multiple_net_peformance = []

    for i in range(repeats):
        dcomp_loss, dcomp = MatrixDecomposition.factorize_matrix(R, lf, iterations=50000)
        mnn_loss = MatrixFactorizationMNN.factorize_matrix(R, lf, 4, iterations=50000)
        snn_loss = MatrixFactorizationSNN.factorize_matrix(R, lf, 8, iterations=50000)

        dcomp_peformance.append(dcomp_loss)
        single_net_peformance.append(snn_loss)
        multiple_net_peformance.append(mnn_loss)

        print(dcomp_loss)
        print(mnn_loss)
        print(snn_loss)

    return dcomp_peformance, single_net_peformance, multiple_net_peformance
    

def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0



A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

zero_out(R, 50)

results = run_tests(R, 5, 10)

print(reults[0], "\n")
print(reults[1], "\n")
print(reults[2], "\n")
