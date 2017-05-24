import tensorflow as tf
import numpy as np

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

R_pre = np.empty(R.shape).tolist()
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        row = R[i,:]
        col = R[:,j]
        c = np.concatenate([row, col])
        
        R_pre[i][j] = c

# Get the entered positions of R
# so we can generate a sparse tensor
indicies = []
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i,j] > 0:
            indicies.append([i,j])

O = tf.constant(R)
O_pre = tf.constant(np.array(R_pre))
idx = tf.constant(indicies, dtype='int64')

theta = init_network(O.shape[0] + O.shape[1], [6,1])
apply_theta = lambda x: apply_network(theta, x)

O_hat = tf.reduce_max(tf.map_fn(apply_theta, O_pre), reduction_indices=2)

error = tf.pow(tf.subtract(O, O_hat), 2.0)
sparse_error = tf.SparseTensor(idx, tf.sqrt(tf.gather_nd(error, idx)), tf.shape(error, out_type=tf.int64))
loss = tf.sparse_reduce_sum(sparse_error)

train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(loss))
    
    for i in range(10000):
        session.run(train_op)
        
    print(session.run(loss))
    
