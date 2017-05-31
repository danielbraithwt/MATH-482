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

def factorize_matrix(R, lf, hs, iterations=40000):
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

    theta = init_network(R.shape[1], [hs, lf])
    fi = init_network(R.shape[0], [hs, lf])

    apply_theta = lambda x: apply_network(theta, tf.transpose(tf.expand_dims(x, 1)))[0]
    apply_fi = lambda x: apply_network(fi, tf.transpose(tf.expand_dims(x, 1)))[0]

    U_lf = tf.map_fn(apply_theta, O)
    M_lf = tf.transpose(tf.map_fn(apply_fi, tf.transpose(O)))

    O_hat = tf.matmul(U_lf, M_lf)

    error = tf.pow(tf.subtract(O, O_hat), 2.0)
    sparse_error = tf.SparseTensor(idx, tf.sqrt(tf.gather_nd(error, idx)), tf.shape(error, out_type=tf.int64))
    loss = tf.sparse_reduce_sum(sparse_error)

    train_op_theta = tf.train.GradientDescentOptimizer(0.000001).minimize(loss, var_list=[theta])
    train_op_fi = tf.train.GradientDescentOptimizer(0.000001).minimize(loss, var_list=[fi])

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            session.run(train_op_theta)
            session.run(train_op_fi)

        final_loss = session.run(loss)
        
    return final_loss

def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0

A = np.random.randint(1, 6, size=(10, 8))
B = np.random.randint(1, 6, size=(9, 8))
R = np.dot(A, np.transpose(B))

zero_out(R, 50)

print(factorize_matrix(R, 5, 10) )
