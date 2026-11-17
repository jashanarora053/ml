import numpy as np 
import pandas as pd 

df1 = pd.read_csv(r"C:\Users\omen\Downloads\Dataset_for_Tuesday_slot_Set_1\ic272_exam_bikes.csv")
df2 = pd.read_csv(r"C:\Users\omen\Downloads\Dataset_for_Tuesday_slot_Set_1\ic272_exam_placement.csv")
df3 = pd.read_csv(r"C:\Users\omen\Downloads\Dataset_for_Tuesday_slot_Set_1\ic272_exam_sensors.csv")
df4 = pd.read_csv(r"C:\Users\omen\Downloads\Dataset_for_Tuesday_slot_Set_1\ic272_exam_towers.csv")

df4.head()

DF4 is for K means clusterring 

Simple K means 

def pairwise_distances(A,B):
    diff = A[:,None,:] - B[None, :,:]
    return np.sqrt((diff**2).sum(axis = 2))
    

def rand_init(X,k,seed = 0):
    idx = np.random.default_rng(seed).choice(len(X), k, replace = False)
    return X[idx]
    

def assign_centroids(X, centroids):
    dist = pairwise_distances(X, centroids)
    labels = dist.argmin(axis = 1)
    return labels

def update_centroids(X, labels, centroids):
    new_centroids = centroids.copy()
    for i in range(len(centroids)):
        X_ = X[labels == i]
        if (len(X_) > 0):
            new_centroids[i] = X[labels == i].mean(axis = 0)
    return new_centroids

def distorsion(X, labels, centroids):
    J = 0
    for i in range(len(centroids)):
        J += ((centroids[i] - X[labels == i]) ** 2).sum()
    return J

def kmeans(X, init_centroids, n_iters = 100):
    centroids = init_centroids
    for i in range(n_iters):
        labels = assign_centroids(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)
        
        if (np.allclose(new_centroids, centroids)) :
            break
        centroids = new_centroids
    labels = assign_centroids(X, new_centroids)
    J = distorsion(X, labels, new_centroids)
    return new_centroids, labels, J, i + 1

X = df4.to_numpy()
out = {}
for sd in range(10):
    init_cent = rand_init(X, 6, sd)
    centroids, labels, j, it = kmeans(X, init_cent)
    out[sd] = {"lab" : labels, "cent" : centroids, "j" : j, "it" : it}

best = min(out, key = lambda s : out[s]["j"])
worst = max(out, key = lambda s : out[s]["j"])

import matplotlib.pyplot as plt
x = df4[df4.columns[0]].to_numpy()
y = df4[df4.columns[1]].to_numpy()
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].scatter(x,y)
ax[0].scatter(out[worst]["cent"][:,0], out[worst]["cent"][:,1])

ax[1].scatter(x,y)
ax[1].scatter(out[best]["cent"][:,0], out[best]["cent"][:,1])


def best_of_kmeans(X, k, n_init = 10):
    best = None
    for sd in range(n_init):
        init_cent = rand_init(X, k,sd)
        cent, lab, j, it = kmeans(X, init_cent)

        if best is None or j < best[2]:
            best = (cent, lab, j, it)
    return best

Ks = list(range(1, 11))
fig, axes = plt.subplots(figsize = (5,5))
axes.plot(Ks, [best_of_kmeans(X,k)[2] for k in Ks], marker = 'o')

def kmeans_pp_init(X, k, seed = 0):
    n = len(X)
    rng = np.random.default_rng(seed)
    id1 = rng.integers(n)
    init_cent = X[id1]
    centroids = []
    centroids.append(init_cent)

    for i in range(1,k):
        C = np.array(centroids)
        D2 = pairwise_distances(X,C).min(axis = 1) ** 2
        probs = D2 / D2.sum()

        idx = rng.choice(n, p = probs)
        centroids.append(X[idx])

    return np.array(centroids, dtype=float)

cents = kmeans_pp_init(X, 5)
fig, axes = plt.subplots(figsize = (5,5))
axes.scatter(X[:,0], X[:,1])
axes.scatter(cents[:,0], cents[:,1])
