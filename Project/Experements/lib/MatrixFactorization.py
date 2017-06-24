import tensorflow as tf
import numpy as np
import scipy.stats as stats
import math
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import MovieLensData

def factorize_matrix(R, Tr, Ts, bias, bias_user, bias_movie, lf, lr=0.0001, lam=1, iterations=10000):
    #print(bias_user)
    #print(bias_movie)
    
    # Get the entered positions of R
    # so we can generate a sparse tensor
    train_indicies = []
    test_indicies = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            #if Tr[i,j] > 0:
            #    train_indicies[i,j] = 1.0
            #if Ts[i,j] > 0:
            #    test_indicies[i,j] = 1.0
            if Tr[i,j] > 0:
                train_indicies.append(i * R.shape[1] + j)
            if Ts[i,j] > 0:
                test_indicies.append(i * R.shape[1] + j)

    R = tf.constant(R, dtype="float64")
    idx = tf.placeholder('int64', [None])
    num = tf.placeholder('float64')
    #mask = tf.placeholder('float64', [None, None])

    u_b = tf.Variable(bias_user)
    m_b = tf.Variable(bias_movie)

    U = tf.Variable(np.random.rand(R.shape[0], lf))
    M = tf.Variable(np.transpose(np.random.rand(R.shape[1], lf)))

    U_w_bias = U#tf.concat([U, u_b, tf.ones((R.shape[0],1), dtype="float64")], 1)
    M_w_bias = M#tf.concat([M, tf.ones((1, R.shape[1]), dtype="float64"), m_b], 0)

    R_hat = tf.matmul(U_w_bias, M_w_bias)
    R_hat_w_bias = R_hat#tf.add(R_hat, bias)

    R_present = tf.gather(tf.reshape(R, [-1]), idx)
    R_hat_present = tf.gather(tf.reshape(R_hat_w_bias, [-1]), idx)

    diff = tf.subtract(R_present, R_hat_present)
    cost_error = tf.reduce_sum(tf.square(diff))

    bias_reg = tf.add(tf.reduce_sum(tf.square(bias_user)), tf.reduce_sum(tf.square(bias_movie)))
    latent_reg = tf.add(tf.reduce_sum(tf.square(U)), tf.reduce_sum(tf.square(M)))

    cost_regularizer = tf.multiply(latent_reg, lam, name="regularize")#tf.multiply(tf.add(bias_reg, latent_reg), lam, name="regularize")#

    loss = cost_error #+ cost_regularizer

    #error = tf.pow(tf.subtract(R, R_hat), 2)#tf.add(tf.abs(tf.subtract(R, R_hat)), lam * (U_reg + M_reg)) 
    #sparse_error = tf.multiply(error, mask)#tf.sparse_tensor_to_dense(tf.SparseTensor(idx, tf.gather_nd(error, idx), tf.shape(error, out_type=tf.int64)))
    #loss = tf.reduce_mean(tf.pow(sparse_error, 2))#tf.norm(sparse_error, ord='fro')#tf.div(tf.sparse_reduce_sum(sparse_error), num)
    #minimize = loss + lam * (U_reg + M_reg)

    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    #train_op_U = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[U])
    #train_op_M = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[M])

    model = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(model)
        #print(session.run(U_w_bias))
        #print(session.run(M_w_bias))
        #print(session.run(R_hat))
        for i in range(iterations):
            session.run(train_op, feed_dict={idx: train_indicies, num:len(train_indicies)})
            #session.run(train_op_U, feed_dict={idx: train_indicies, num:len(train_indicies)})
            #session.run(train_op_M, feed_dict={idx: train_indicies, num:len(train_indicies)})

            if i % 100 == 0:
                
                tl = session.run(cost_error, feed_dict={idx: test_indicies, num:len(test_indicies)})
                l = session.run([cost_error], feed_dict={idx: train_indicies, num:len(train_indicies)})
                print("Loss: {}".format(l))
                print("Test Loss: {}\n".format(tl))
        
        train_loss_final = session.run(cost_error, feed_dict={idx: train_indicies, num:len(train_indicies)})
        test_loss_final = session.run(cost_error, feed_dict={idx: test_indicies, num:len(test_indicies)})
        U_final = session.run(U)
        M_final = session.run(M)

    return train_loss_final, test_loss_final, (U_final, M_final)


def zero_out(R, num):
    idx = list(np.ndindex(R.shape))
    np.random.shuffle(idx)

    for i in range(num):
        R[idx[i]] = 0

def conf_interval(pop):
    z = z_critical = stats.norm.ppf(q = 0.95)
    moe = z * (pop.std()/math.sqrt(len(pop)))

    return (pop.mean() - moe, pop.mean() + moe)


np.random.seed(1234)

A = np.random.rand(20, 15)
B = np.random.rand(23, 15)
R = np.dot(A, np.transpose(B))
R_Origonal = np.copy(R)

z = R.shape[0] * R.shape[1] * (1.0/2.0)

zero_out(R, int(z))

R_T = np.zeros(R.shape)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i,j] == 0:
            R_T[i,j] = R_Origonal[i,j]


# Compute Biases
# Avg Rating
mu = R[np.nonzero(R)].mean()

bias_user = []
for i in range(R.shape[0]):
    user = R[i,:]
    bias_user.append(mu - user[np.nonzero(user)].mean())

bias_movie = []
for i in range(R.shape[1]):
    movie = R[:,i]
    bias_movie.append(mu - movie[np.nonzero(movie)].mean())


bias_user = np.expand_dims(np.array(bias_user), 1)
bias_movie = np.transpose(np.expand_dims(np.array(bias_movie), 1))

performance = []
performance_test = []
for i in range(10):
    #train, test, dcomp = factorize_matrix(R_Origonal, R, R_T, mu, bias_user, bias_movie, 15, lr=0.00002, lam=1.5, iterations=200000)
    train, test, dcomp = factorize_matrix(R_Origonal, R, R_T, mu, bias_user, bias_movie, 15, lr=0.0002, lam=1.5, iterations=200000)
    performance.append(train)
    performance_test.append(test)

performance = np.array(performance)
performance_test = np.array(performance_test)
print(performance.mean())
print(conf_interval(performance))

print()
print(performance_test.mean())
print(conf_interval(performance_test))

##m_train = []
##m_test = []
##std_train = []
##std_test = []
##
##for i in range(0, 30, 2):
##    train_peformance = []
##    test_peformance = []
##    for j in range(5):
##        print(j)
##        train, test, dcomp = factorize_matrix(R_Origonal, R, R_T, 8, lr=0.001, lam=i, iterations=500000)
##
##        train_peformance.append(train)
##        test_peformance.append(test)
##
##    train_peformance = np.array(train_peformance)
##    test_peformance = np.array(test_peformance)
##
##    print("Training: ", train_peformance.mean())
##    print("Test: ", test_peformance.mean())
##
##    m_train.append(train_peformance.mean())
##    m_test.append(test_peformance.mean())
##
##    std_train.append(train_peformance.std())
##    std_test.append(test_peformance.std())
##
##x_axis = np.array(range(0,30,2))
##df = np.repeat(10-1, len(range(0,30,2)))
##
##plt.plot(x_axis, m_train, '-o', color='b', label='Training')
##plt.errorbar(x_axis, m_train, yerr=stats.t.ppf(0.95, df)*std_train, color='b')
##
##plt.plot(x_axis, m_test, '-o', color='r', label='Test')
##plt.errorbar(x_axis, m_test, yerr=stats.t.ppf(0.95, df)*std_test, color='r')
##
##plt.ylabel("Accuracy")
##plt.xlabel("Regularization Paramater Value")
##plt.xlim([0, 31])
##plt.legend(loc='best')
##plt.savefig("lamvsperformance.png")

#print("Training CI: ", conf_interval(np.array(train_peformance)))
#print("Test CI: ", conf_interval(np.array(test_peformance)))
