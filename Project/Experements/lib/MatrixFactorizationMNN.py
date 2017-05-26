import tensorflow as tf
import numpy as np

def init_network(inputs, layers):
    network = []
    
    current_in = inputs
    for l in layers:
        layer = tf.Variable(0.02 * (-0.5 + np.random.rand(l, current_in + 1)), dtype='float64')
        current_in = l

        network.append(layer)

    return network


def apply_network(network, inputs):
    current_out = inputs
    for layer in network:
        current_out = tf.concat([tf.expand_dims(np.repeat([1.0], current_out.shape[0]), 1), current_out], axis=1)
        current_out = tf.matmul(current_out, tf.transpose(layer))

    return current_out


def factorize_matrix(R, lf, interval=100, iterations=10001):
    losses = []
    # Get the entered positions of R
    # so we can generate a sparse tensor
    indicies = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > 0:
                indicies.append([i,j])

    O = tf.constant(R, dtype='float64')
    idx = tf.constant(indicies, dtype='int64')

    theta = init_network(O.shape[1], [6,lf])
    fi = init_network(O.shape[0], [6,lf])


    U_lf = apply_network(theta, O)
    M_lf = apply_network(fi, tf.transpose(O))

    O_hat = tf.matmul(U_lf, tf.transpose(M_lf))

    error = tf.pow(tf.subtract(O, O_hat), 2.0)
    sparse_error = tf.SparseTensor(idx, tf.sqrt(tf.gather_nd(error, idx)), tf.shape(error, out_type=tf.int64))
    loss = tf.sparse_reduce_sum(sparse_error)

    train_op_theta = tf.train.GradientDescentOptimizer(0.00001).minimize(loss, var_list=theta)
    train_op_fi = tf.train.GradientDescentOptimizer(0.00001).minimize(loss, var_list=fi)

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            if i % interval == 0:
                l = session.run(loss)
                losses.append(l)
                
            session.run(train_op_theta)
            session.run(train_op_fi)
            
    return losses
