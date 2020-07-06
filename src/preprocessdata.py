import numpy as np
from sklearn import preprocessing


def sorted_coulomb_matrix(X):
    def _realize_(x):
        inds = np.argsort(-np.linalg.norm(x, axis=1))  # sort indices of coulomb matrix
        x = x[inds][:, inds]  # permute rows first, then columns
        x = x.flatten()[triuind]  # flatten and slice
        return x

    return np.array([_realize_(z) for z in X])


def randomized_coulomb_matrix(X, noise=1, seed=234):
    def _realize_(x):
        rs = np.random.RandomState(seed=seed)
        gaussian_error = rs.normal(0, noise, x[0].shape)
        inds = np.argsort(-np.linalg.norm(x, axis=1) + gaussian_error)  # sort indices of coulomb matrix
        x = x[inds][:, inds]  # permute rows first, then columns
        x = x.flatten()[triuind]  # flatten and slice
        return x

    return np.array([_realize_(z) for z in X])


def expand(X, step=1):
    Xexp = []
    for i in range(X.shape[1]):
        for k in np.arange(0, np.max(i) + step, step):
            Xexp += [np.tanh((X[:, i] - k) / step)]
    return np.array(Xexp).T


if __name__ == "__main__":
    X = np.load('../data/coulomb_matrix.npy')
    y = np.load('../data/atomization_energies.npy')

    '''
    triuind = (np.arange(3)[:,np.newaxis] <= np.arange(3)[np.newaxis,:]); triuind
    array([[ True,  True,  True],
       [False,  True,  True],
       [False, False,  True]])
    '''

    triuind = (np.arange(23)[:, np.newaxis] <= np.arange(23)[np.newaxis, :]).flatten()

    X_sorted = sorted_coulomb_matrix(X)
    X_randomized = randomized_coulomb_matrix(X)
    # X_expanded = expand(randomized_coulomb_matrix(X))

    '''
    X_scaled = Zero mean and unit variance

    '''

    X_sorted_scaled = preprocessing.StandardScaler().fit_transform(X_sorted)
    X_randomized_scaled = preprocessing.StandardScaler().fit_transform(X_randomized)
    # X_expanded_scaled = preprocessing.StandardScaler().fit_transform(X_expanded)

    print('X_sorted_scaled Mean:', X_sorted_scaled.mean())
    print('X_sorted_scaled Std Dev:', X_sorted_scaled.std())
    print('X_randomized_scaled Mean:', X_randomized_scaled.mean())
    print('X_randomized_scaled Std Dev:', X_randomized_scaled.std())
    # print('X_expanded_scaled Mean:', X_expanded_scaled.mean())
    # print('X_expanded_scaled Std Dev:', X_expanded_scaled.std())

    '''
    X_minmax_scaled = Max 1 and min 0

    '''

    X_sorted_minmax_scaled = preprocessing.MinMaxScaler().fit_transform(X_sorted)
    X_randomized_minmax_scaled = preprocessing.MinMaxScaler().fit_transform(X_randomized)
    # X_expanded_minmax_scaled = preprocessing.MinMaxScaler().fit_transform(X_expanded)

    print('X_sorted_minmax_scaled max :', X_sorted_minmax_scaled.max())
    print('X_sorted_minmax_scaled min:', X_sorted_minmax_scaled.min())
    print('X_randomized_minmax_scaled max :', X_randomized_minmax_scaled.max())
    print('X_randomized_minmax_scaled min:', X_randomized_minmax_scaled.min())
    # print('X_expanded_minmax_scaled max :', X_expanded_minmax_scaled.max())
    # print('X_expanded_minmax_scaled min:', X_expanded_minmax_scaled.min())

    y = y[0]

    np.save('../input/X_sorted_scaled.npy', X_sorted_scaled)
    np.save('../input/X_sorted_minmax_scaled.npy', X_sorted_minmax_scaled)
    np.save('../input/X_randomized_scaled.npy', X_randomized_scaled)
    np.save('../input/X_randomized_minmax_scaled.npy', X_randomized_minmax_scaled)
    np.save('../input/y.npy', y)
