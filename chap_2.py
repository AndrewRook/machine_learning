import numpy as np
from scipy.spatial import cKDTree

def vectorized_nn(X):
    XXT = np.dot(X,X.T)
    Xii = XXT.diagonal()

    D = Xii - 2* XXT + Xii[:,np.newaxis]

    return np.argsort(D,axis=1)[:,1]

def kdtree_nn(X,return_dists = False):
    kdt = cKDTree(X)
    dists,neighbors = kdt.query(X,k=2)
    output = neighbors[:,1]
    if return_dists:
        output = (neighbors[:,1],dists[:,1])
    return output
