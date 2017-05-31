import tensorflow as tf
import numpy as np

def factorize_matrix(R, lf, iterations=10000):
    # Get the entered positions of R
    # so we can generate a sparse tensor
    indicies = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > 0:
                indicies.append([i,j])

    R = tf.constant(R, dtype="float64")
    idx = tf.constant(indicies, dtype='int64')

    U = tf.Variable(np.random.rand(R.shape[0], lf))
    M = tf.Variable(np.transpose(np.random.rand(R.shape[1], lf)))

    R_hat = tf.matmul(U, M)

    error = tf.pow(tf.subtract(R, R_hat), 2.0)
    sparse_error = tf.SparseTensor(idx, tf.sqrt(tf.gather_nd(error, idx)), tf.shape(error, out_type=tf.int64))
    loss = tf.sparse_reduce_sum(sparse_error)

    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    model = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(model)

        for i in range(iterations):    
            session.run(train_op)
        
        loss_final = session.run(loss)
        R_hat_final = session.run(R_hat)
        U_final = session.run(U)
        M_final = session.run(M)

    return loss_final, (U_final, M_final)
