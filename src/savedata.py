import numpy as np
import scipy.io

if __name__ == "__main__":

    data = scipy.io.loadmat('../qm7.mat')
    
    '''
    Write data from a Matlab .mat file to numpy arrays.
    X = Coulomb Matrix
    y = Atomization energies
    R = Cartesian Coordinates
    P = Folds 
    '''
    
    X = data['X']
    y = data['T']
    R = data['R']
    P = data['P']

    np.save('../data/coulomb_matrix.npy', X)
    np.save('../data/atomization_energies.npy', y)
    np.save('../data/cartesian_coordinates.npy', R)
    np.save('../data/folds.npy', P)




