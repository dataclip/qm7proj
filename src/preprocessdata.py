import numpy as np
from sklearn import preprocessing


if __name__ == "__main__":

    X = np.load('../data/coulomb_matrix.npy')
    y = np.load('../data/atomization_energies.npy')

    '''
    triuind = (np.arange(3)[:,np.newaxis] <= np.arange(3)[np.newaxis,:]); triuind
    array([[ True,  True,  True],
       [False,  True,  True],
       [False, False,  True]])
    '''

    triuind = (np.arange(23)[:,np.newaxis] <= np.arange(23)[np.newaxis,:]).flatten()

    def sorted_coulomb_matrix(X):
        def _realize_(x):
            inds = np.argsort(-np.linalg.norm(x, axis=1)) # sort indices of coulomb matrix
            x = x[inds][:,inds] # permute rows first, then columns
            x = x.flatten()[triuind] # flaten and slice
            return x
        return np.array([_realize_(z) for z in X])

    X = sorted_coulomb_matrix(X)

    '''
    X_scaled = Zero mean and unit variance

    '''

    X_scaled = preprocessing.StandardScaler().fit_transform(X)

    print('X_scaled Mean:', X_scaled.mean())
    print('X_scaled Std Dev:', X_scaled.std())

    '''
    X_minmax_scaled = Max 1 and min 0

    '''

    X_minmax_scaled = preprocessing.MinMaxScaler().fit_transform(X)

    print('X_minmax_scaled max :', X_minmax_scaled.max())
    print('X_minmax_scaled min:', X_minmax_scaled.min())

    y = y[0]

    np.save('../input/X_scaled.npy', X_scaled)
    np.save('../input/X_minmax_scaled.npy', X_minmax_scaled)
    np.save('../input/y.npy', y)


