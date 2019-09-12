import numpy as np

def euclidean_distance(a, b):
    if len(a) != len(b):
        print('Different lenght')
        return False
    # distance = 0
    # for i in range(len(a)):
    return np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))
    # return np.sqrt(distance)


from math import sqrt, pow
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return sqrt(distance)

    # return np.sqrt(np.sum([np.power((num_a-num_b), 2) for num_a, num_b in zip(a, b)]))

def k_neighbours(np_arrays, k=1):
    arr_size = len(np_arrays)
    all_dist = []
    for i in range(arr_size-1):
        print(i)
        for j in range(i+1, arr_size):
            # all_dist.append(euclideanDistance(np_arrays[i], np_arrays[j], len(np_arrays[j])))
            all_dist.append(euclidean_distance(np_arrays[i], np_arrays[j]))
            # all_dist.append(np.linalg.norm(np_arrays[i]-np_arrays[j]))
        # print(all_dist)

    np.array(all_dist)
    return all_dist[np.int(np.amin(all_dist))]


    # return all_dist

with open('/home/lucas/Downloads/test.dat', 'rt') as f:
    tmp_test = [x.strip().split(' ') for x in f.readlines()]
    X_test = []
    y_test = []
    for x in tmp_test :
        X_test.append([float(y.split(':')[-1]) for y in x[1:]])
        y_test.append(int(x[0]))

with open('/home/lucas/Downloads/train.dat', 'rt') as f:
    tmp_train = [x.strip().split(' ') for x in f.readlines()]
    X_train = []
    y_train = []
    for x in tmp_train:
        X_train.append([float(y.split(':')[-1]) for y in x[1:]])
        y_train.append(int(x[0]))


import pandas as pd
len(X_train)
pd_train = pd.DataFrame([x for x in X_train])
pd_train.describe()
pd_train.iloc[0, :]


np_arrays = pd_train.iloc[:1000, :].values
np_arrays.shape
a = k_neighbours(pd_train.iloc[:1000, :].values)
b = k_neighbours(pd_train.iloc[:1000, :].values)


a,b = pd_train.iloc[0, :], pd_train.iloc[1, :]
np.all(np.isclose(a,b))

# np.clo

np.linalg.norm(np.array([1,1,1])+np.array([1,1,1]))

np.sqrt(12)

test_result = euclidean_distance(pd_train.iloc[0, :].values, pd_train.iloc[1, :].values)
np_result = np.linalg.norm(pd_train.iloc[0, :].values - pd_train.iloc[1, :].values)

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=5)
knn.fit(np_arrays)
knn.kneighbors(np_arrays)