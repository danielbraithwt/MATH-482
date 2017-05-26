
import sys
sys.path.append('../lib/')

import MatrixFactorization as MatrixDecomposition

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))


LF_start = 1
LF_end = 8
repeat = 5

for 
