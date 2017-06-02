import tensorflow as tf
import math
import sys
sys.path.append('../lib/')

import MatrixFactorization as MatrixDecomposition
import MatrixFactorizationSNN
import MatrixFactorizationMNN
import numpy as np

K = 10
lf = 5

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

def create_grid(space):
    grid = []

    # Set minimums
    count = np.repeat(0, len(space)).tolist()

    finished = False
    while not finished:
        entry = []
        for i in range(len(count)):
            entry.append(space[i][count[i]])
        
        grid.append(entry)

        finished = True
        for i in range(len(count)):
            count[i] += 1
            if count[i] >= len(space[i]):
                count[i] = 0
            else:
                finished = False
                break

    return grid

def split_data(R, K):
    partitions = []
    
    idx = list(np.ndindex(R.shape))
    idx_non_zero = []
    for i in idx:
        if not R[i] == 0:
            idx_non_zero.append(i)

    np.random.shuffle(idx_non_zero)
    
    sub_size = int(len(idx_non_zero)/K)
    for i in range(0, len(idx_non_zero), sub_size):
        Tr = np.copy(R)
        Ts = np.copy(R)

        for j in range(0, len(idx_non_zero)):
            if j >= i and j < (i+sub_size):
                Tr[idx_non_zero[j]] = 0
            else:
                Ts[idx_non_zero[j]] = 0

        partitions.append((Tr,Ts))

    return partitions
        
    

matrix_decomp_grid = create_grid([[0.1,0.01,0.001,0.0001,0.00001], [1, 5, 10, 15, 20, 25, 30], [5000, 10000, 15000]])
best_config = None
best_loss = math.inf

data_sets = split_data(R, K)

for config in matrix_decomp_grid:
    print("Config: ", config)
    total_loss = 0
    for d in data_sets:
        training_loss, test_loss, dcomp = MatrixDecomposition.factorize_matrix(R, d[0], d[1], lf, config[0], config[1], config[2])
        total_loss += test_loss
        print("\tTraining Loss: ", training_loss)
        print("\tTest Loss: ", test_loss)

    loss = float(total_loss)/float(len(data_sets))
    print("Loss: ", loss, "\n")
    if loss < best_loss:
        best_loss = loss
        best_config = config

print()
print("-- BEST --")
print("Config: ", best_config)
print("Loss: ", loss)
    
