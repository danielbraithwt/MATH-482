import tensorflow as tf
import math
import sys
sys.path.append('../lib/')

import MatrixFactorization as MatrixDecomposition
import MatrixFactorizationSNN
import MatrixFactorizationMNN
import numpy as np

K = 2
lf = 8

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0

zero_out(R, 50)

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
        
    

#matrix_decomp_grid = create_grid([[0.01,0.001,0.0001], [0.001, 0.002, 0.003], [10000, 20000, 30000]])
#matrix_decomp_grid = create_grid([[0.0004], [0.03], [60000]])
mnn_grid = create_grid([[3, 5, 8, 12], [0.00001, 0.000001, 0.0000001], [5000, 10000, 15000]])
snn_grid = create_grid([[3, 5, 8, 12], [0.00001, 0.000001, 0.0000001], [5000, 10000, 15000]])


##best_config_dcomp = None
##best_test_loss_dcomp = math.inf
##best_training_loss_dcomp = math.inf

best_config_mnn = None
best_test_loss_mnn = math.inf
best_training_loss_mnn = math.inf

best_config_snn = None
best_test_loss_snn = math.inf
best_training_loss_snn = math.inf

data_sets = split_data(R, K)

##for config in matrix_decomp_grid:
##    print("Config: ", config)
##    total_test_loss = 0
##    total_training_loss = 0
##    for d in data_sets:
##        training_loss, test_loss, dcomp = MatrixDecomposition.factorize_matrix(R, d[0], d[1], lf, config[0], config[1], config[2])
##        total_test_loss += test_loss
##        total_training_loss += training_loss
##        print("\tTraining Loss: ", training_loss)
##        print("\tTest Loss: ", test_loss)
##
##    test_loss = float(total_test_loss)/float(len(data_sets))
##    training_loss = float(total_training_loss)/float(len(data_sets))
##    print("Test Loss: ", test_loss)
##    print("Training Loss: ", training_loss, "\n")
##    if test_loss < best_test_loss_dcomp:
##        best_test_loss_dcomp = test_loss
##        best_training_loss_dcomp = training_loss
##        best_config_dcomp = config


for config in mnn_grid:
    print("Config: ", config)
    total_test_loss = 0
    total_training_loss = 0
    for d in data_sets:
        training_loss, test_loss = MatrixFactorizationMNN.factorize_matrix(R, d[0], d[1], lf, config[0], config[1], config[2])
        total_test_loss += test_loss
        total_training_loss += training_loss
        print("\tTraining Loss: ", training_loss)
        print("\tTest Loss: ", test_loss)

    test_loss = float(total_test_loss)/float(len(data_sets))
    training_loss = float(total_training_loss)/float(len(data_sets))
    print("Test Loss: ", test_loss)
    print("Training Loss: ", training_loss, "\n")
    if test_loss < best_test_loss_mnn:
        best_test_loss_mnn = test_loss
        best_training_loss_mnn = training_loss
        best_config_mnn = config

for config in snn_grid:
    print("Config: ", config)
    total_test_loss = 0
    total_training_loss = 0
    for d in data_sets:
        training_loss, test_loss = MatrixFactorizationSNN.factorize_matrix(R, d[0], d[1], lf, config[0], config[1], config[2])
        total_test_loss += test_loss
        total_training_loss += training_loss
        print("\tTraining Loss: ", training_loss)
        print("\tTest Loss: ", test_loss)

    test_loss = float(total_test_loss)/float(len(data_sets))
    training_loss = float(total_training_loss)/float(len(data_sets))
    print("Test Loss: ", test_loss)
    print("Training Loss: ", training_loss, "\n")
    if test_loss < best_test_loss_snn:
        best_test_loss_snn = test_loss
        best_training_loss_snn = training_loss
        best_config_snn = config

print()
print("-- BEST --")
##print("DCOMP")
##print("Config: ", best_config_dcomp)
##print("Test Loss: ", best_test_loss_dcomp)
##print("Training Loss: ", best_training_loss_dcomp)
print("MNN")
print("Config: ", best_config_mnn)
print("Test Loss: ", best_test_loss_mnn)
print("Training Loss: ", best_training_loss_mnn)
print("SNN")
print("Config: ", best_config_snn)
print("Test Loss: ", best_test_loss_snn)
print("Training Loss: ", best_training_loss_snn)
    
