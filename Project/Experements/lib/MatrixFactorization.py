import tensorflow as tf
import numpy as np

def factorize_matrix(R, Tr, Ts, lf, lr=0.0001, lam=1, iterations=10000):
    # Get the entered positions of R
    # so we can generate a sparse tensor
    train_indicies = []
    test_indicies = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if Tr[i,j] > 0:
                train_indicies.append([i,j])
            if Ts[i,j] > 0:
                test_indicies.append([i,j])

    R = tf.constant(R, dtype="float64")
    idx = tf.placeholder('int64', [None, 2])
    num = tf.placeholder('float64')

    U = tf.Variable(np.random.rand(R.shape[0], lf))
    M = tf.Variable(np.transpose(np.random.rand(R.shape[1], lf)))

    R_hat = tf.matmul(U, M)

    error = tf.abs(tf.subtract(R, R_hat))
    sparse_error = tf.SparseTensor(idx, tf.gather_nd(error, idx), tf.shape(error, out_type=tf.int64))
    loss = tf.div(tf.sparse_reduce_sum(sparse_error), num)
    minimize = loss + lam * (tf.norm(U) + tf.norm(M))
    
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(minimize)

    model = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(model)

        for i in range(iterations):    
            session.run(train_op, feed_dict={idx: train_indicies, num:len(train_indicies)})
        
        train_loss_final = session.run(loss, feed_dict={idx: train_indicies, num:len(train_indicies)})
        test_loss_final = session.run(loss, feed_dict={idx: test_indicies, num:len(test_indicies)})
        U_final = session.run(U)
        M_final = session.run(M)

    return train_loss_final, test_loss_final, (U_final, M_final)
