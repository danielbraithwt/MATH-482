import tensorflow as tf
import numpy as np

def init_network(inputs, layers):
    network = []
    
    current_in = inputs
    for l in layers:
        layer = tf.Variable((-0.5 + (np.random.rand(l, current_in + 1))), dtype='float64')
        current_in = l

        network.append(layer)

    return network


def apply_network(network, inputs):
    current_out = inputs
    for layer in network:
        current_out = tf.concat([tf.expand_dims(np.repeat([1.0], current_out.shape[0]), 1), current_out], axis=1)
        current_out = tf.matmul(current_out, tf.transpose(layer))

    return current_out

def sigmoid(tensor):
    return 1.0/(1.0 + tf.exp(-tensor))

def factorize_matrix(R, Tr, Ts, lf, hs, lr=0.0001, iterations=10000):
    losses = []
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

    O = tf.constant(R, dtype='float64')
    idx = tf.placeholder('int64', [None, 2])
    num = tf.placeholder('float64')

    theta = init_network(R.shape[1], [hs, lf])
    fi = init_network(R.shape[0], [hs, lf])

    apply_theta = lambda x: apply_network(theta, tf.transpose(tf.expand_dims(x, 1)))[0]
    apply_fi = lambda x: apply_network(fi, tf.transpose(tf.expand_dims(x, 1)))[0]

    U_lf = tf.map_fn(apply_theta, O)
    M_lf = tf.transpose(tf.map_fn(apply_fi, tf.transpose(O)))

    O_hat = tf.matmul(U_lf, M_lf)

    error = tf.abs(tf.subtract(O, O_hat))
    sparse_error = tf.SparseTensor(idx, tf.gather_nd(error, idx), tf.shape(error, out_type=tf.int64))
    loss = tf.div(tf.sparse_reduce_sum(sparse_error), num)

    train_op_theta = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[theta])
    train_op_fi = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[fi])

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            session.run(train_op_theta, feed_dict={idx: train_indicies, num:len(train_indicies)})
            session.run(train_op_fi, feed_dict={idx: train_indicies, num:len(train_indicies)})
        
        train_loss_final = session.run(loss, feed_dict={idx: train_indicies, num:len(train_indicies)})
        test_loss_final = session.run(loss, feed_dict={idx: test_indicies, num:len(test_indicies)})
        
    return train_loss_final, test_loss_final
