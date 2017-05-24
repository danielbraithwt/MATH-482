import tensorflow as tf
import numpy as np

lf = 3

def init_network(inputs, layers):
    network = []
    
    current_in = inputs
    for l in layers:
        layer = tf.Variable(-0.5 + np.random.rand(l, current_in + 1))
        current_in = l

        network.append(layer)

    return network


def apply_network(network, inputs):
    current_out = inputs
    for layer in network:
        current_out = tf.concat([tf.expand_dims(np.repeat([1.0], current_out.shape[0]), 1), current_out], axis=1)
        current_out = tf.matmul(current_out, tf.transpose(layer))

    return current_out

A = np.array(
[
    [1.0,2.0,1.0,4.0],
    [2.0,4.0,3.0,1.0],
    [1.0,5.0,4.0,2.0],
    [2.0,4.0,4.0,3.0]
])

B = np.array(
[
    [1.0,2.0,1.0,4.0],
    [1.0,4.0,6.0,1.0],
    [4.0,5.0,4.0,2.0],
    [1.0,2.0,4.0,1.0],
])

R = np.dot(A, B)

# Get the entered positions of R
# so we can generate a sparse tensor
indicies = []
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i,j] > 0:
            indicies.append([i,j])

O = tf.constant(R)
idx = tf.constant(indicies, dtype='int64')

theta = init_network(O.shape[0], [6,lf])
fi = init_network(O.shape[1], [6,lf])


U_lf = apply_network(theta, O)
M_lf = apply_network(fi, tf.transpose(O))

O_hat = tf.matmul(U_lf, tf.transpose(M_lf))

error = tf.pow(tf.subtract(O, O_hat), 2.0)
sparse_error = tf.SparseTensor(idx, tf.sqrt(tf.gather_nd(error, idx)), tf.shape(error, out_type=tf.int64))
loss = tf.sparse_reduce_sum(sparse_error)

train_op_theta = tf.train.GradientDescentOptimizer(0.000001).minimize(loss, var_list=theta)
train_op_fi = tf.train.GradientDescentOptimizer(0.000001).minimize(loss, var_list=fi)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(loss))
    
    for i in range(10000):
        session.run(train_op_theta)
        session.run(train_op_fi)
        
    print(session.run(loss))
    
