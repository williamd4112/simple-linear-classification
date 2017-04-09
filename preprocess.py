import numpy as np
import numpy.matlib
import csv

from tqdm import *

from numpy import vstack,array
from numpy.random import rand

class Preprocessor(object):
    def pca(self, X, k):
        '''
            Calculate covariance matrix
        '''
        cov_mat = np.cov(X.T)
        
        '''
            Calculate eigen vector
        '''
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        
        '''
            Sort by eigen value
        '''
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_pairs = np.array([eig_pairs[i][1] for i in xrange(k)])
        
        return X.dot(eig_pairs.T), eig_pairs
       

    def normalize(self, X):
        obs = X
        std_dev = np.std(obs, axis=0)
        return obs / std_dev, std_dev

if __name__ == '__main__':
    from util import load_categorical_dataset
    import sys
    X, Y = load_categorical_dataset(sys.argv[1], sys.argv[2])

    X_normal = Preprocessor().normalize(X, 255.0)
    pca_w = Preprocessor().pca(X_normal, k=2)
    
    X_phi = X_normal * pca_w.T
    print X_phi
