import tensorflow as tf
import numpy as np

def factorise_matrix(R, lf):
    # Get the entered positions of R
    # so we can generate a sparse tensor
    indicies = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > 0:
                indicies.append([i,j])


    latent_variables = tf.constant(lf)
    idx = tf.constant(indicies, dtype='int64')
    O = tf.constant(R)

    U = tf.Variable(0.5 * np.random.rand(O.shape[0], latent_variables))
    M = tf.Variable(0.5 * np.transpose(np.random.rand(O.shape[1], latent_variables)))

    O_hat = tf.matmul(U, M)

    error = tf.pow(tf.subtract(O, O_hat), 2.0)
    sparse_error = tf.SparseTensor(idx, tf.gather_nd(error, idx), tf.shape(error, out_type=tf.int64))
    loss = tf.sparse_reduce_sum(sparse_error)

    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        print(session.run(loss))

        for i in range(1000):
            session.run(train_op)
        
        loss_final = session.run(loss))
        O_hat_final = session.run(O_hat)
        U_final = session.run(U)
        M_final = session.run(M)


    return O_hat_final, (U_final, M_final), loss_final



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

print("-- R --")
print(R)

R[1,2] = 0
R[2,1] = 0
R[0,0] = 0

print("-- R0 --")
print(R)
