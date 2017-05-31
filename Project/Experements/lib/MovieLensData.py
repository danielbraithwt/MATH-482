import numpy as np

num_users = 943
num_movies = 1682

def read_data():
    R = np.zeros((num_users, num_movies))
    f = open('/home/braithdani/MATH-482/Project/Experements/lib/Data/ml-100k/u.data', 'r')

    l = f.readline()
    while l:
        d = l.split('\t')

        u = int(d[0]) - 1
        m = int(d[1]) - 1
        r = int(d[2])

        R[u,m] = r
        l = f.readline()

    f.close()

    return R
